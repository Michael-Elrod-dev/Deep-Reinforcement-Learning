import pygame
import torch
import numpy as np
from logger import Logger
from collections import deque
from minigrid.core.world_object import Agent
from minigrid.core.mission import MissionSpace
from minigrid.core.minigrid import MultiGridEnv
from dqn import Network
"""
    drop errors
    gaussian errors
    robust learning
"""
def preprocess_state(image):
    # This selects the second element across all rows and columns
    image_matrix = image[:, :, 1]
    return image_matrix.reshape(-1)

def test_dqn(env, agents, load_policy, network, logger, max_steps):
    if load_policy: network.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    eps = 0
    total_steps = 0
    n_episodes = 10
    total_scores = []
    steps_per_episode_history = []

    for episode in range(n_episodes):
        initial_observations, _ = env.reset()
        scores = [0 for _ in agents]
        steps_per_episode = 0

        # Preprocess the initial state for each agent and store
        states = [preprocess_state(obs['image']) for obs in initial_observations]
        agent_pos = [tuple(np.array(agent.cur_pos)) for agent in agents]
        agent_dirs = [agent.dir for agent in agents]
        states = [np.concatenate([np.array(pos), np.array([dir]), state.flatten()]) for pos, dir, state in zip(agent_pos, agent_dirs, states)]

        for step in range(max_steps):
            total_steps += 1
            steps_per_episode += 1
            actions = network.action(states, eps)
            next_states, rewards, done = env.step(actions)

            # Apply preprocessing and positions to the state representation
            next_states_flat = [preprocess_state(state['image']) for state in next_states]
            agent_pos = [tuple(agent.cur_pos) for agent in agents]
            agent_dirs = [agent.dir for agent in agents]
            next_states_flat = [np.concatenate([np.array(pos), np.array([dir]), state.flatten()]) for pos, dir, state in zip(agent_pos, agent_dirs, next_states_flat)]
            
            for index, agent in enumerate(agents):
                scores[index] += rewards[index]

            states = next_states_flat
            if done:
                break

        episode_total_score = sum(scores)
        total_scores.append(episode_total_score)
        steps_per_episode_history.append(steps_per_episode)
        logger.log_test_metrics(total_steps, episode+1, eps, episode_total_score, np.mean(total_scores), steps_per_episode, np.mean(steps_per_episode_history))

        print(f'Steps: {total_steps}\tEpisode: {episode+1}\tScore: {episode_total_score}\tAverage Score: {np.mean(total_scores):.2f}')
        
    return total_scores

def dqn(env, agents, network, logger, n_steps, max_steps, eps_start, eps_end, eps_decay):
    saved = False
    current_episode = 0
    total_steps = 0
    eps = eps_start
    total_scores = deque(maxlen=100)
    steps_per_episode_history = deque(maxlen=100)

    while total_steps < n_steps:
        current_episode += 1
        steps_per_episode = 0
        scores = [0 for _ in agents]
        initial_observations, _ = env.reset()

        # Preprocess the initial state for each agent and store
        states = [preprocess_state(obs['image']) for obs in initial_observations]
        agent_pos = [tuple(np.array(agent.cur_pos)) for agent in agents]
        agent_dirs = [agent.dir for agent in agents]
        states = [np.concatenate([np.array(pos), np.array([dir]), state.flatten()]) for pos, dir, state in zip(agent_pos, agent_dirs, states)]

        network.update_beta((total_steps-1)/(n_steps-1))
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = True
                        while paused:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    paused = False
            total_steps += 1
            steps_per_episode += 1
            actions = network.action(states, eps)
            next_states, rewards, done = env.step(actions)
            
            # Apply preprocessing and positions to the state representation
            next_states_flat = [preprocess_state(state['image']) for state in next_states]
            agent_pos = [tuple(agent.cur_pos) for agent in agents]
            agent_dirs = [agent.dir for agent in agents]
            next_states_flat = [np.concatenate([np.array(pos), np.array([dir]), state.flatten()]) for pos, dir, state in zip(agent_pos, agent_dirs, next_states_flat)]
            
            for index, agent in enumerate(agents):
                print(type(states[index]), type(actions[index]), type(rewards[index]), type(next_states_flat[index]), type(done))
                network.step(states[index], actions[index], rewards[index], next_states_flat[index], done)
                scores[index] += rewards[index]

            states = next_states_flat  
            if eps > eps_end:
                eps -= eps_decay
            else:
                eps = eps_end
                
            if done:
                break

        episode_total_score = sum(scores)
        total_scores.append(episode_total_score)
        steps_per_episode_history.append(steps_per_episode)
        # logger.log_metrics(total_steps, current_episode, eps, episode_total_score, np.mean(total_scores), steps_per_episode, np.mean(steps_per_episode_history))
        # logger.log_agent_metrics(current_episode, scores)

        print(f'\rSteps: {total_steps}\tEpisode: {current_episode}\tEpsilon: {eps:.2f}\tScore: {episode_total_score}\tAverage Score: {np.mean(total_scores):.2f}', end="")
        if current_episode % 100 == 0:
            print(f'\rSteps: {total_steps}\tEpisode: {current_episode}\tEpsilon: {eps:.2f}\tScore: {episode_total_score}\tAverage Score: {np.mean(total_scores):.2f}')
            torch.save(network.qnetwork_local.state_dict(), 'checkpoint.pth')

    torch.save(network.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

def calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage):
    effective_steps = n_steps * eps_percentage
    decrement_per_step = (eps_start - eps_end) / effective_steps
    return decrement_per_step

def main():
    load_policy = False
    n_steps = 2000000
    max_steps = 450
    num_collectables = 50
    num_agents = 3
    action_size = 3
    eps_start = 1.0
    eps_end = 0.01
    eps_percentage = 0.98
    eps_decay = calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage)
    seed = 0
    Double_DQN = True
    Priority_Replay_Paras = [0.5, 0.5, 0.5]

    agents = [Agent(id=i, direction=3) for i in range(num_agents)]
    mission_space = MissionSpace(mission_func=lambda: "Get the green ball.", ordered_placeholders=None)
    env = MultiGridEnv(
        num_collectables=num_collectables,
        mission_space=mission_space,
        grid_size=21,
        agents=agents,
        agent_view_size=7
        )
    
    state_size = (env.agent_view_size ** 2) + 3
    network = Network(state_size, action_size, seed, Double_DQN, Priority_Replay_Paras)
    logger = Logger(num_agents, action_size, state_size, eps_start, eps_end, eps_decay, n_steps, max_steps)
    if not load_policy:
        _ = dqn(env, agents, network, logger, n_steps, max_steps, eps_start, eps_end, eps_decay)
    _ = test_dqn(env, agents, load_policy, network, logger, max_steps)

    logger.close()
    env.close()

if __name__ == '__main__':
    main()