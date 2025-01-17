from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar, List, Tuple
from copy import deepcopy as copy

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.RewardType import RewardType
from minigrid.core.world_object import WorldObj, Agent, Ball, Wall, Point
from minigrid.core.constants import COLOR_NAMES, TILE_PIXELS, OBJECT_TO_IDX

T = TypeVar("T")

# Multi-agent base class
class MultiGridEnv(gym.Env):
    """
    2D grid world game multi-agent environment
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(
        self,
        num_collectables,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = True,
        agents:List[Agent]=None,
        render_mode: str | None = 'human',
        screen_size: int | None = 640,
        highlight: bool = True,
        highlight_agents: List[int] = None,
        tile_size: int = TILE_PIXELS,
        agent_view_size: int = 7,
        partial_obs=True,
        actions_set=Actions,
        window_name="Custom Multi-Agent MiniGrid",
    ):
        # set the agents
        assert len(agents) > 0, "Must have at least one agent"
        self.agents = agents
        for agent in self.agents:
            agent.mission = mission_space.sample()

        self.missions = {agent.id: agent.mission for agent in self.agents}
        self.agent_view_size = agent_view_size

        # set environment parameters
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Set the actions
        self.actions = actions_set
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_space = spaces.Tuple(tuple([self.action_space] * len(self.agents)))

        # Set the observation
        self.partial_obs = partial_obs
        if partial_obs:
            image_observation_space = spaces.Box(
                low=0, high=255, shape=(self.agent_view_size, self.agent_view_size, 9), dtype='uint8'
            )
        else:
            image_observation_space = spaces.Box(
                low=0, high=255, shape=(self.width, self.height, 9), dtype='uint8'
            )
        observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "mission": mission_space,
            }
        )
        self.observation_space = spaces.Tuple(tuple(observation_space for _ in range(len(self.agents))))

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls
        self.grid = Grid(width, height)

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        if highlight_agents is None:
            highlight_agents = [agent.id for agent in self.agents]
        self.highlight_agents = highlight_agents
        self.tile_size = tile_size

        self.window_name = window_name
        self.mission_text = ""
        self.verbose_text = ""

        self.num_collectables = num_collectables

    def reset(self, *, seed=None, options=None)-> tuple[ObsType, dict[str, Any]]:
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height) 
        
        # check the agents in the environment
        for id, agent in enumerate(self.agents):
            assert agent.init_pos is not None
            assert agent.cur_pos is not None
            assert agent.dir is not None
            assert id == agent.id, f"Agent id {agent.id} specified in the environment does not match the id {id} of the agent in the list of agents"
        
        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            raise NotImplementedError

        return obs, {}
   
    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property	
    def steps_remaining(self):	
        return self.max_steps - self.step_count	
     
    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                agent = self.grid.get_agent(i, j)
                if agent is not None:
                    output += AGENT_DIR_TO_STR[agent.dir] + agent.color[0].upper()
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < self.grid.height - 1:
                output += "\n"

        return output

    def place_obj(
        self,
        obj: WorldObj | Agent | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
        layer: str = 'objects'
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        assert layer in ['objects', 'agents'], f'Invalid layer {layer}'

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            if self.grid.get_agent(*pos) != None:
                #don't place the object on top of another agent
                continue

            # # Don't place the object on top of another object
            # if self.grid.get(*pos) != None and layer == 'objects':
            #     #don't place the object on top of another object
            #     continue

            # Don't place the object can't overlap
            if self.grid.get(*pos) and not self.grid.get(*pos).can_overlap() and layer == 'agents':
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break
        
        if layer == 'objects':
            self.grid.set(*pos, obj)
        elif layer == 'agents':
            self.grid.set_agent(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    @abstractmethod
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place agents at the bottom of the grid with maximum space between them
        num_agents = len(self.agents)
        for index, agent in enumerate(self.agents):
            num_agents = len(self.agents)
            
            if num_agents > 1:
                interval = (width - 3) // (num_agents - 1)
                x_position = 1 + interval * index
            else:
                x_position = (width - 1) // 2
            y_position = height - 2
            agent.init_pos = (x_position, y_position)
            agent.cur_pos = (x_position, y_position)
            
            agent.dir = 3
            self.grid.set_agent(x_position, y_position, agent)

        # Place collectible objects randomly in the grid
        for _ in range(self.num_collectables):
            obj = Ball(color="green")
            while True:
                x, y = self._rand_pos(0, width - 1, 0, height - 2)
                if not self.grid.get(x, y):
                    self.grid.set(x, y, obj)
                    break

    @abstractmethod
    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        pass
    
    def _handle_overlap(self, i, rewards, fwd_pos, fwd_cell):
        if isinstance(fwd_cell, Ball) and fwd_cell.color == "green":
            fwd_cell.color = "blue"
            self.agents[i].num_collected += 1
            rewards[i] += 5

        self.grid.set_agent(*fwd_pos, self.agents[i])
        self.grid.set_agent(*self.agents[i].cur_pos, None)
        self.agents[i].cur_pos = fwd_pos

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell.can_pickup() and self.agents[i].carrying is None:
            self.agents[i].carrying = copy(fwd_cell)
            self.grid.set(*fwd_pos, None)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying is not None:
            self.grid.set(*fwd_pos, copy(self.agents[i].carrying))
            self.agents[i].carrying = None

    def _reward(self, agent_id, discount_factor=0.9, max_reward=1):
        """
        Compute the reward to be given upon success
        """
        return max_reward - discount_factor * (self.step_count / self.max_steps)
        
    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """
        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """
        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """
        return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """
        lst = list(iterable)
        assert num_elems <= len(lst)
        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self)-> str:
        """
        Generate a random color name (string)
        """
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh) -> Tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """
        return (
            self.np_random.integers(xLow, xHigh),
            self.np_random.integers(yLow, yHigh)
        )

    def put_obj(self, obj:WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set_agent(i, j, obj) if obj.encode()[0] == OBJECT_TO_IDX["agent"] else self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self, 
        agent_id: int,
        top:Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
        agent_dir: int = -1,
    ):
        """
        Place the agent at an empty position in the grid
        """
        assert agent_id < len(self.agents), f'Invalid agent id {agent_id}'
        self.agents[agent_id].cur_pos = None
        pos = self.place_obj(self.agents[agent_id], top, size, max_tries=max_tries, reject_fn=reject_fn,layer='agents')
        self.agents[agent_id].dir = 3

        return pos

    def agent_sees(self, agent: Agent, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """
        coordinates = agent.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, actions: List[ActType]) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert len(actions) == len(self.agents)
        
        self.step_count += 1
        done = False

        rewards = np.zeros(len(self.agents))
        terminated = False
        truncated = False

        order = np.random.permutation(len(self.agents))

        for i in order:
            reward = 0
            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started:
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Check if the forward position is valid (not a wall and within bounds)
            fwd_cell = self.grid.get(*fwd_pos)
            fwd_agent = self.grid.get_agent(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir += 1
                if self.agents[i].dir > 3:
                    self.agents[i].dir -= 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if isinstance(fwd_cell, Wall):
                    rewards[i] += -1
                elif 0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height:
                    if fwd_cell is None and fwd_agent is None:
                        self.grid.set_agent(*fwd_pos, self.agents[i])
                        self.grid.set_agent(*self.agents[i].cur_pos, None)
                        self.agents[i].cur_pos = fwd_pos
                    elif fwd_cell and fwd_cell.can_overlap():
                        self._handle_overlap(i, rewards, fwd_pos, fwd_cell)

            # Pick up an object
            elif actions[i] == self.actions.pickup:
                if fwd_cell and not fwd_agent:
                    self._handle_pickup(i, rewards, fwd_pos, fwd_cell) # TODO: add pickup reward

            # Drop an object
            elif actions[i] == self.actions.drop:
                if not (fwd_cell or fwd_agent):
                    self._handle_drop(i, rewards, fwd_pos, fwd_cell) #TODO: add drop reward

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        total_collected = sum(agent.num_collected for agent in self.agents)
        if total_collected >= self.num_collectables:
            done = True

        if self.render_mode == "human":
            self.render()

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            raise NotImplementedError
        
        return obs, rewards, done

    def gen_obs_grid(self, agent_view_size: int | None = None):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        grids = []
        vis_masks = []
        if agent_view_size is None:
            agent_view_size = self.agent_view_size

        for agent in self.agents:
            topX, topY, botX, botY = agent.get_view_exts(agent_view_size)
            agent_view_size = agent_view_size or self.agent_view_size

            grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)
            for i in range(agent.dir + 1):
                grid = grid.rotate_left()
            
            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(
                    agent_pos=(agent_view_size // 2, agent_view_size // 2), agent_id=agent.id)
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)
            
            grids.append(grid)
            vis_masks.append(vis_mask)
        
        return grids, vis_masks

    def gen_obs(self):
        """	
        Generate the agent's view (partially observable, low-resolution encoding)	
        """	
        
        grids, vis_masks = self.gen_obs_grid()

        assert isinstance(self.missions, dict), "mission must be a dictionary indicating the mission for agents"
        
        image = tuple([grid.encode(vis_mask) for grid, vis_mask in zip(grids, vis_masks)])
        obs = [(lambda i:{
                'image': image[i],
                'mission': self.missions[i],
            })(i) for i in range(len(self.agents))]
        
        return tuple(obs)

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """
        grid, vis_mask = Grid.decode(obs)
        # Render the whole grid
        img = grid.render(
            tile_size,
            highlight_mask=vis_mask
        )
        return img
    
    def render(self):
        """
        Render the whole-grid human view
        """
        img = self.get_full_render(self.highlight, self.tile_size)

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
                pygame.display.set_caption("MA_minigrid")
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
            text = self.verbose_text
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            
            # Render the additional line of text at the top
            top_text = self.mission_text
            top_text_rect = font.get_rect(top_text, size=font_size)
            top_text_rect.center = bg.get_rect().center
            top_text_rect.y = font_size * 0.5
            font.render_to(bg, top_text_rect, top_text, size=font_size)
            
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            #self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_masks = self.gen_obs_grid()

        # Mask of which cells to highlight
        highlight_masks = np.zeros((self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for i in self.highlight_agents:
            a = self.agents[i]
            f_vec = a.dir_vec
            r_vec = a.right_vec
            top_left = (
                a.cur_pos
                + f_vec * (a.view_size // 2)
                - r_vec * (a.view_size // 2)
            )
            for vis_j in range(0, a.view_size):
                for vis_i in range(0, a.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_masks[i][vis_i, vis_j]:
                        continue
                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue
                    
                    # Mark this cell to be highlighted
                    highlight_masks[abs_i, abs_j] = True
        
        # Render the whole grid
        img = self.grid.render(
            tile_size,
            highlight_mask=highlight_masks if highlight else None
        )
        return img
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

        