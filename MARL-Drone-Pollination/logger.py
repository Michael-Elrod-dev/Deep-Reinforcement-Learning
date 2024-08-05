import wandb

class Logger():
    def __init__(self, num_agents, action_size, state_size, eps_start, eps_end, eps_decay, n_steps, max_steps_per_episode):
        wandb.init(project='Multi-Agent DQN', entity='elrod-michael95', name=f'DQN: {num_agents} Agents')
        wandb.config.update({
            "num_agents": num_agents,
            "action_size": action_size,
            "state_size": state_size,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "n_steps": n_steps,
            "max_steps_per_episode": max_steps_per_episode,
        })

    def log_metrics(self, steps, episodes, epsilon, reward, average_reward, steps_per_episode, average_steps_per_episode):
        wandb.log({
            "Steps per Episode": steps_per_episode,
            "Average Steps per Episode": average_steps_per_episode,  # New metric logged
            "Episodes": episodes,
            "Epsilon": epsilon,
            "Reward": reward,
            "Average Reward": average_reward
        })

    def log_agent_metrics(self, episodes, agents_reward):
        # Prepare a dictionary for batch logging
        log_data = {"Episodes": episodes}
        
        # Add each agent's score to the dictionary with a unique namespace
        for agent_id, reward in enumerate(agents_reward):
            metric_name = f"Agent_{agent_id}/Reward"
            log_data[metric_name] = reward

        wandb.log(log_data)

    def log_test_metrics(self, steps, episodes, epsilon, reward, average_reward, steps_per_episode, average_steps_per_episode):
        wandb.log({
            "Test/Steps per Episode": steps_per_episode,
            "Test/Average Steps per Episode": average_steps_per_episode,
            "Test/Episodes": episodes,
            "Test/Epsilon": epsilon,
            "Test/Reward": reward,
            "Test/Average Reward": average_reward
        })
        
    def close(self):
        wandb.finish()
