# # import gym as old_gym
# import gymnasium as gym
# import shimmy
# from panda_gym.envs.panda_tasks.panda_push_safe import PandaPushSafeEnv
# import panda_gym
# import time
# # from panda_gym.envs.panda_tasks.panda_push_safe import PandaPushSafeEnv
# # old_env = PandaPushSafeEnv(render_mode='human')
# # compat_env = EnvCompatibility(old_env)

# old_env = PandaPushSafeEnv(render_mode='human')
# # old_env = old_gym.make("PandaPushSafe-v2", render_mode='human', apply_api_compatibility=True)
# # env = gym.make(env=env)

# class GymToGymnasiumAdapter(gym.Env):
#     def __init__(self, old_env):
#         self.old_env = old_env
#         self.observation_space = old_env.observation_space
#         self.action_space = old_env.action_space
#         self.render_mode = getattr(old_env, 'render_mode', 'human')

#     def reset(self, seed=None, options=None):
#         if seed is not None and hasattr(self.old_env, 'seed'):
#             self.old_env.seed(seed)
#         obs = self.old_env.reset()
#         return obs, {}

#     def step(self, action):
#         obs, reward, done, info = self.old_env.step(action)
#         # If the environment already returns terminated/truncated, adjust accordingly
#         # But based on your earlier errors, it returns a single 'done'
#         terminated = done
#         truncated = False
#         return obs, reward, terminated, truncated, info

#     def render(self):
#         return self.old_env.render()

#     def close(self):
#         self.old_env.close()

# env = GymToGymnasiumAdapter(old_env)
# # env = gym.make("GymV26Environment-v0", env=old_env)
# # env = shimmy.GymV21Environment(old_env)
# # env = gym.make("GymV21Environment-v0", env=old_env) # or V26


import gymnasium as gym
from panda_gym.envs.panda_tasks.panda_push_safe import PandaPushSafeEnv
from panda_gym.envs.panda_tasks.gym2gymnasium_adapter import GymToGymnasiumAdapter


# Create and wrap
old_env = PandaPushSafeEnv(render_mode='human')
env = GymToGymnasiumAdapter(old_env)

# # Use the environment
# obs, info = env.reset()
# for i in range(10):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Step {i}: reward={reward}")
#     if terminated or truncated:
#         break
# env.close()


obs, info = env.reset()
done = False
mdp = []
while not done:
    transition = []
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    cost = info.get("cost", 0)
    print(cost)
    transition =[obs, reward, done, info["cost"]]
    mdp.append(transition)
    # env.render(mode='human')


env.close()
