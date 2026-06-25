import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def select_action(self, state):
        return self.choose_action(state)

    def update_q_value(self, state, action, reward, next_state):
        best_next_value = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
