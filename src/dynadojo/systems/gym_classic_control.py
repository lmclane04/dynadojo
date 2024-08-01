import numpy as np
from ..abstractions import AbstractSystem

if not hasattr(np, 'float_'):
    np.float_ = np.float64

import gymnasium as gym


def get_all_envs():
    classic_control_envs = []
    for env_id, spec in gym.envs.registry.items():
        # Check if 'classic_control' is part of the entry_point string
        entry_point = getattr(spec, 'entry_point', '')
        if isinstance(entry_point, str) and 'classic_control' in entry_point:
            classic_control_envs.append(env_id)
    return classic_control_envs



class GymClassicControlSystem(AbstractSystem):
    all_envs = get_all_envs()

    def __init__(self, latent_dim, embed_dim, env_name: str, seed=None):
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.env_name = env_name
        all_envs = get_all_envs()
        
        if env_name not in all_envs:
            raise ValueError(f"Environment name '{env_name}' not found in Open AI Gym Classic Control Systems library.")
        
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.embed_dim = self.env.observation_space.shape[0]
        np.random.seed(self.seed)

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        low, high = self.env.observation_space.low, self.env.observation_space.high
        if in_dist:
            init_conds = np.random.uniform(low, high, size=(n, len(low)))
        else:
            init_conds = np.random.uniform(low - 10, high + 10, size=(n, len(low)))
        
        return init_conds


    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False):
        n = init_conds.shape[0]
        data = np.zeros((n, timesteps, self._embed_dim))

        for i in range(n):
            _, _ = self.env.reset()
            self.env.state = init_conds[i]
            
            observations = []
            trajectory = [self.env.state]
            for t in range(timesteps):
                if control is not None and not np.array_equal(control, np.zeros_like(control)):
                    action = control[i, t]
                else:
                    action = self.env.action_space.sample()

                next_observation, reward, done, truncated, info = self.env.step(action)

                observations.append(next_observation)

                trajectory.append(next_observation)
                

                if done or truncated:
                    next_observation, _ = self.env.reset()
                    trajectory.append(next_observation)

            trajectory = np.array(trajectory)
            if len(trajectory) > timesteps:
                trajectory = trajectory[:timesteps]
            elif len(trajectory) < timesteps:
                trajectory = np.pad(trajectory, ((0, timesteps - len(trajectory)), (0, 0)), mode='edge')

            data[i] = np.array(trajectory)
        
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self._embed_dim