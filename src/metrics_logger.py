class MetricsLogger:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.steps = []

    def log_episode(self, episode, reward, steps):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)

    def clear(self):
        self.episodes.clear()
        self.rewards.clear()
        self.steps.clear()
