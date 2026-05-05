import gym
import panda_gym
import time
env = gym.make("PandaStack3-v2", render=True)
# PandaStackPyramid
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
    # time.sleep(4)

env.close()
