import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from environment import EmptyEnv
from dqn_agent import Agent

def test_dqn(env, load_policy, n_steps=100):
    if load_policy:
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    scores = []
    score = 0
    for _ in range(3):
        initial_observation, _ = env.reset()
        # Ensure consistent preprocessing of state before testing
        state = preprocess_state(initial_observation['image'])
        for step in range(n_steps):
            eps = 0
            action = agent.act(state, eps)

            # Take action and observe the result
            next_state, reward, done = env.step(action)
            next_state = preprocess_state(next_state['image'])

            score += reward
            state = next_state

            if done:
                break
        scores.append(score)
    return scores

def preprocess_state(image):
    # This selects the second element across all rows and columns
    image_matrix = image[:, :, 1]
    return image_matrix.reshape(-1)

def dqn(env, state_size, action_size, n_episodes=1000, max_steps=350, eps_start=1.0, eps_end=0.01, eps_decay=0.997, 
        Double_DQN=True, Priority_Replay_Paras=[0.5, 0.5, 0.5]):
    global agent
    agent = Agent(state_size, action_size, seed=0, Double_DQN=Double_DQN, Priority_Replay_Paras=Priority_Replay_Paras)
    scores = []
    saved = False
    eps = eps_start
    scores_window = deque(maxlen=100)

    for episode in range(1, n_episodes + 1):
        initial_observation, _ = env.reset()
        # Apply preprocessing to the initial state
        state = preprocess_state(initial_observation['image'])
        agent.update_beta((episode-1)/(n_episodes-1))
        score = 0
        for step in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)

            # Apply preprocessing to the state representation
            next_state_flat = preprocess_state(next_state['image'])
            # agent_pos = np.array(env.agent_pos)
            # next_state_flat = np.concatenate([agent_pos, next_state_flat])

            
            agent.step(state, action, reward, next_state_flat, done)
            
            state = next_state_flat
            #print(state)
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(episode, eps, score, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(episode, eps, score, np.mean(scores_window)))

        if np.mean(scores_window) >= ((env.num_collectibles-1)*5):
            if not saved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
                saved = True
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')            
    if not saved:
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

def plot(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    indices = np.arange(0, len(scores), 100)
    scores_subset = scores[::100]
    plt.plot(indices, scores_subset, marker='o')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    load_policy = False
    env = EmptyEnv(render_mode="human")

    if not load_policy:
        original_shape = env.observation_space['image'].shape
        state_size = np.prod(original_shape[:-1])
        action_size = 3

        scores = dqn(env, state_size, action_size)
        plot(scores)

    test_scores = test_dqn(env, load_policy, n_steps=150)
    plot(test_scores)

    env.close()

if __name__ == '__main__':
    main()
