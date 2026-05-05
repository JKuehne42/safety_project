import gym
import panda_gym
import time

env = gym.make("PandaPickAndPlaceSafe-v2", render=True)

obs = env.reset()
done = False
mdp = []
while not done:
    transition = []
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cost = info["cost"]
    transition = [obs, reward, done, info["cost"]]
    mdp.append(transition)
    env.render(mode='human')


env.close()
