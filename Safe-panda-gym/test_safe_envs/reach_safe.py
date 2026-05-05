import gym
import panda_gym
import time
# env = gym.make("PandaReach-v2", render=True)
env = gym.make("PandaReachSafe-v2", render=True)
obs_dim = env.observation_space.shape
# print(obs_dim)

obs = env.reset()
# print(obs.shape)
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cost = info['cost']
    # print(obs.shape)
    env.render(mode='human')
    print(cost)
    time.sleep(2)


env.close()
