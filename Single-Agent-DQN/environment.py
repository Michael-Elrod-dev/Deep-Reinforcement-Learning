from __future__ import annotations
import numpy as np
import pygame
from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Floor
from minigrid.utils.rendering import fill_coords, point_in_rect

class EmptyEnv(MiniGridEnv):
    def __init__(
        self,
        size=21,
        agent_start_dir=0,
        agent_view_size = 9,
        max_steps: int | None = None,
        **kwargs,
    ):
        middle = size // 2
        self.agent_start_pos = (middle, middle)
        self.agent_start_dir = agent_start_dir

        self.num_collectibles = 20
        self.on_blue = 0
        self.collected_items = 0
        self.step_count = 0
        self.max_steps = 500
        self.goal_positions = [(2, 1), (6, 5), (1, 7), (6, 7), (1, 3), (7, 3), (1, 4), (4, 3), (1, 2), (5, 3),
                            (1, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            agent_view_size = agent_view_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def step(self, action):
        # Perform the action and get the result
        next_state, reward, done, _, _ = super().step(action)

        # Increment step count
        self.step_count += 1

        # Get the current position of the agent
        current_cell = self.grid.get(*self.agent_pos)

        # If there is a Floor object at the agent's position
        if isinstance(current_cell, Floor):
            if current_cell.color == 'green':
                # Change the floor's color to blue to indicate it's been visited
                current_cell.color = 'blue'
                self.grid.set(*self.agent_pos, current_cell)
                self.collected_items += 1
                reward += 5  # Reward for visiting for the first time
            elif current_cell.color == 'blue':
                reward -= 0.2  # Penalty for revisiting
                pass

        # Check if all items are collected
        if self.collected_items >= self.num_collectibles:
            done = True
            reward += 10
        
        return next_state, reward, done
    
    def reset(self,*,seed: int | None = None,options: dict[str, Any] | None = None,) -> tuple[ObsType, dict[str, Any]]:
            super().reset(seed=seed)

            # Reinitialize episode-specific variables
            self.agent_pos = (-1, -1)
            self.agent_dir = -1

            # Generate a new random grid at the start of each episode
            self._gen_grid(self.width, self.height)

            # These fields should be defined by _gen_grid
            assert (
                self.agent_pos >= (0, 0)
                if isinstance(self.agent_pos, tuple)
                else all(self.agent_pos >= 0) and self.agent_dir >= 0
            )

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos)
            assert start_cell is None or start_cell.can_overlap()

            # Item picked up, being carried, initially nothing
            self.carrying = None

            # Step count since episode start
            self.step_count = 0
            self.collected_items = 0

            if self.render_mode == "human":
                self.render()

            # Return first observation
            obs = self.gen_obs()

            return obs, {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for pos in self.goal_positions:
            self.place_obj(Floor(), pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Collect Green Squares"

    def get_view_exts(self, agent_view_size=None):
        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - agent_view_size + 1
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def gen_obs_grid(self, agent_view_size=None):
        if agent_view_size is None:
            agent_view_size = self.agent_view_size

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        # Rotate the grid according to the agent's direction to maintain orientation.
        for i in range(self.agent_dir):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(agent_view_size // 2, agent_view_size // 2))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs
    
    def get_pov_render(self, tile_size):

        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size // 2),
            agent_dir=3,
            highlight_mask=vis_mask,
        )
        print("hello")
        return img
    
    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            # Remove frame rate limit
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img


class Floor(WorldObj):
    def __init__(self, color: str = "green"):
        super().__init__("floor", color)
        self.color = 'green'

    def can_overlap(self):
        return True

    def render(self, img):
        # Convert the color name to its RGB value
        color = COLORS[self.color]  # Assuming 'green' is in the COLORS dictionary
        fill_coords(img, point_in_rect(0, 1, 0, 1), color)
        