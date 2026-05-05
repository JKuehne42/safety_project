import gym
import panda_gym
import time
env = gym.make("PandaBuildL-v2", render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
    # time.sleep(4)

env.close()
