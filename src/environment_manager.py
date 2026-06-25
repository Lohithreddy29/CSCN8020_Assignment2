import gymnasium as gym


class TaxiEnvironmentManager:
    def __init__(self, env_name="Taxi-v3"):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
