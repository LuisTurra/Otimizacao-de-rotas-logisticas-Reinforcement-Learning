import numpy as np
import random
from typing import Tuple, Dict, List

class SupplyChainEnv:
    def __init__(self, locations: List[Dict], depot: Dict, max_steps: int = 50):
        self.locations = locations
        self.original_demands = np.array([loc['demand'] for loc in locations])  
        self.depot = depot
        self.n_locations = len(locations)
        self.action_space = self.n_locations + 1
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> Tuple[int, tuple]:
        self.current_pos = self.depot
        self.current_idx = -1
        self.remaining_demand = np.array([loc['demand'] for loc in self.locations])
        self.steps = 0
        self.total_reward = 0
        self.last_distance = 0
        self.last_delivery_step = 0
        self.done = False
        self.route_positions = [self.depot]  
        return self._get_state()

    def _get_state(self) -> Tuple[int, tuple]:
        return (self.current_idx, tuple(self.remaining_demand))

    def _valid_actions(self) -> List[int]:
        pending = [i for i in range(self.n_locations) if self.remaining_demand[i] > 0]
        return pending + [self.n_locations]

    def step(self, action: int) -> Tuple[Tuple[int, tuple], float, bool, Dict]:
        self.steps += 1
        done = False
        reward = 0
        delivery_made = False
        prev_pos = self.current_pos.copy()
        dist = 0.0  

        if action == self.n_locations:  
            dist = self._haversine(prev_pos, self.depot)  
            self.last_distance += dist
            self.current_idx = -1
            self.current_pos = self.depot
            self.route_positions.append(self.depot)
            if np.all(self.remaining_demand <= 0) or self.steps >= self.max_steps:
                reward += 100
                done = True
            else:
                reward -= 50
        else:
            if self.remaining_demand[action] <= 0:
                reward -= 20
                return self._get_state(), reward, done, {'prev_pos': prev_pos}

            
            dist = self._haversine(self.current_pos, self.locations[action])
            self.last_distance += dist
            self.current_pos = self.locations[action]
            self.current_idx = action
            self.remaining_demand[action] = 0
            self.route_positions.append(self.locations[action])
            delivery_made = True
            self.last_delivery_step = self.steps

            reward = -dist * 0.1 + 30

            
            if self.steps - self.last_delivery_step > 10:
                reward -= 10
                self.current_idx = -1
                self.current_pos = self.depot
                self.route_positions.append(self.depot)
                done = np.all(self.remaining_demand <= 0)

        # Timeout
        if self.steps >= self.max_steps:
            reward -= 100
            done = True

        self.total_reward += reward
        self.done = done
        return self._get_state(), reward, done, {
            'dist': dist,
            'delivery_made': delivery_made,
            'prev_pos': prev_pos,
            'action': action,
            'demand_original': self.original_demands[action] if action < self.n_locations else 0
        }

    def _haversine(self, loc1: Dict, loc2: Dict) -> float:
        R = 6371
        lat1, lon1 = np.radians([loc1['lat'], loc1['lon']])
        lat2, lon2 = np.radians([loc2['lat'], loc2['lon']])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def render(self):
        print(f"Pos: {self.current_pos}, Idx: {self.current_idx}, Remaining: {self.remaining_demand}, Steps: {self.steps}, Reward: {self.total_reward:.2f}")