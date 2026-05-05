import gym
import panda_gym
import time

env = gym.make("PandaStackSafe-v2", render=True)

obs = env.reset()
done = False
mdp = []
while not done:
    transition = []
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cost = info.get("cost", 0)
    print(cost)
    transition =[obs, reward, done, info["cost"]]
    mdp.append(transition)
    env.render(mode='human')


env.close()
