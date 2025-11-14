import pickle
import random
from collections import defaultdict
from typing import Tuple, List

class QLearningAgent:
    def __init__(self, n_actions: int, env=None, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)

    def choose_action(self, state: Tuple[int, tuple]) -> int:
        valid_actions = self._get_valid_actions()
        if not valid_actions:
            return self.n_actions - 1  

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        
        q_vals = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_vals[a] for a in valid_actions)
        candidates = [a for a in valid_actions if q_vals[a] == max_q]
        return random.choice(candidates)

    def _get_valid_actions(self) -> List[int]:
        if self.env is not None and hasattr(self.env, '_valid_actions'):
            try:
                return self.env._valid_actions()
            except:
                pass
        return list(range(self.n_actions))  

    def learn(self, state, action, reward, next_state, done):
        current = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            next_qs = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            target = reward + self.gamma * max(next_qs)
        self.q_table[(state, action)] += self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = defaultdict(float, pickle.load(f))