class QLearningTrainer:
    def __init__(self, env, agent, episodes, logger, experiment_name="baseline"):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.logger = logger
        self.experiment_name = experiment_name

    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.agent.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            self.logger.log_episode(episode, total_reward, steps)

        return self.logger
