import gymnasium as gym

class GymToGymnasiumAdapter(gym.Env):
    def __init__(self, old_env):
        self.old_env = old_env
        self.observation_space = old_env.observation_space
        self.action_space = old_env.action_space
        self.render_mode = getattr(old_env, 'render_mode', 'human')

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self.old_env, 'seed'):
            self.old_env.seed(seed)
        obs = self.old_env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.old_env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.old_env.render()

    def close(self):
        self.old_env.close()