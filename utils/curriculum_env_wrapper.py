import gymnasium as gym
import numpy as np

class CurriculumLunarLander(gym.Wrapper):
    def __init__(self, env, easy_episodes=200):
        super().__init__(env)
        self.easy_episodes = easy_episodes
        self.episode_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1

        if self.episode_count <= self.easy_episodes:
            # Make initial state "easier": smaller velocities, more central
            obs = np.array(obs, dtype=np.float32)
            obs[0] *= 0.5          # x closer to center
            obs[1] = max(obs[1], 0.8)  # y higher up / stable
            obs[2] *= 0.2          # vx
            obs[3] *= 0.2          # vy
            obs[4] *= 0.5          # angle
            obs[5] *= 0.5          # angular velocity
        return obs, info
