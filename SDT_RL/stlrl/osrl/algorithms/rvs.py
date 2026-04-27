import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import SquashedGaussianMLPActor, MLPGaussianActor
from stlcg.stlcg import STL_Formula


class RvS(nn.Module):
    """
    Reinforcement Learning via Supervised Learning (RvS)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list, optional): List of integers specifying the sizes 
            of the layers in the actor network.
        episode_len (int, optional): Maximum length of an episode.
        device (str, optional): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.episode_len = episode_len
        self.device = device

        # self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
        #                                       self.a_hidden_sizes,
        #                                       nn.ReLU).to(self.device)
        self.actor = MLPGaussianActor(self.state_dim, self.action_dim, 
                                      -self.max_action, self.max_action, 
                                      self.a_hidden_sizes, nn.ReLU).to(self.device)

    def actor_loss(self, observations, actions):
        # _, _, pi_dist = self.actor.forward(observations, False, False, True)
        # logp_pi = pi_dist.log_prob(actions).sum(axis=-1)
        # logp_pi -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        pi_dist, _, _ = self.actor.forward(observations)
        logp_pi = pi_dist.log_prob(actions).sum(axis=-1)
        loss_actor = -logp_pi.mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}
        return loss_actor, stats_actor

    def setup_optimizers(self, actor_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def act(self, obs):
        '''
        Given a single obs, return the action.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        # act, _ = self.actor.forward(obs, True)
        _, act, _ = self.actor.forward(obs, None, True)
        act = act.data.numpy() if self.device == "cpu" else act.data.cpu().numpy()
        return np.squeeze(act, axis=0)


class RvSTrainer:
    """
    RvS Trainer
    
    Args:
        model (RvS): The RvS model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        rvs_mode (str): specify rvs mode
        cost_limit (int): Upper limit on the cost per episode.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: RvS,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            rvs_mode: str = "RvS",
            cost_limit: int = 10,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            # relabel_cost: bool = False,
            device="cpu"):

        self.model = model
        self.logger = logger
        self.env = env
        self.device = device
        self.rvs_mode = rvs_mode
        self.cost_limit = cost_limit
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        # self.relabel_cost = relabel_cost
        self.model.setup_optimizers(actor_lr)

    def set_target_cost(self, target_cost):
        self.cost_limit = target_cost

    def train_one_step(self, observations, actions):
        """
        Trains the model by updating the actor.
        """
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
        self.logger.store(**stats_actor)

    def evaluate(self, eval_episodes: int, target_return: float, target_cost: float,
                 eval_specification: STL_Formula=None, task: str=None, threshold: int=5):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        episode_rew_rob, episode_cost_rob = [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost, rew_rob, cost_rob = self.rollout(target_return, target_cost,
                                                      eval_specification, task, threshold)
            episode_rets.append(epi_ret); episode_lens.append(epi_len); episode_costs.append(epi_cost)
            episode_rew_rob.append(rew_rob); episode_cost_rob.append(cost_rob)
        episode_rew_rob = None
        episode_cost_rob = None if eval_specification is None else np.array(episode_cost_rob)
        safe_rate = np.sum(np.array(episode_costs) <= 0.0) / eval_episodes
        
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens), episode_rew_rob, episode_cost_rob, safe_rate

    @torch.no_grad()
    def rollout(self,
        target_return: float,
        target_cost: float,
        eval_specification: STL_Formula=None,
        task: str=None,
        threshold: int=5):
        """
        Evaluates the performance of the model on a single episode.
        """
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        obs, info = self.env.reset()
        cur_ret, cur_cost = target_return, target_cost
        consecutive_ones = 0
        rew_rob, cost_rob = None, None

        if self.rvs_mode == "RvS-R":
            obs = np.append(obs, cur_ret)
        elif self.rvs_mode == "RvS-RC":
            obs = np.append(obs, [cur_ret, cur_cost])
        elif self.rvs_mode == "RvS-RCR":
            if "Circle" in task:
                states_xa = np.array([obs[0]])
                xa = torch.tensor(states_xa, dtype=torch.float).reshape([1, states_xa.shape[0], 1])
                cost_inputs = ((xa, xa), ((xa, xa)))
            if "Run" in task:
                xdot = 2; ydot = 3
                if "Drone" in task or "Ant" in task:
                    xdot += 1; ydot += 1
                states_ya = np.array([obs[1]])
                states_va = np.array([np.hypot(obs[xdot], obs[ydot])])
                ya = torch.tensor(states_ya, dtype=torch.float).reshape([1, states_ya.shape[0], 1])
                va = torch.tensor(states_va, dtype=torch.float).reshape([1, states_va.shape[0], 1])
                cost_inputs = ((ya, ya), (va, va))

            cost_rob = eval_specification.robustness(cost_inputs)[0, 0, 0]
            obs = np.append(obs, [cur_ret, target_cost, cost_rob])
            
        for _ in range(self.model.episode_len):
            act = self.model.act(obs)
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            # relabel cost
            if "Circle" in task:
                cost = info["cost"]
                if cost == 1.0:
                    consecutive_ones += 1
                    new_cost = 1 if consecutive_ones > threshold else 0
                else:
                    consecutive_ones = 0
                    new_cost = 0
                cost = new_cost * self.cost_scale
            if "Run" in task:
                cost_bound = info["cost_outside_bounds"]
                cost_vel = info["cost_velocity_violation"]
                if cost_vel == 1.0:
                    consecutive_ones += 1
                    new_cost = 1 if consecutive_ones > threshold else 0
                else:
                    consecutive_ones = 0
                    new_cost = 0
                cost_vel = new_cost
                cost = np.min([1, cost_bound + cost_vel]) * self.cost_scale

            # compute rtg and ctg
            cur_ret -= reward
            cur_cost -= cost
            
            if self.rvs_mode == "RvS-R":
                obs_next = np.append(obs_next, cur_ret)
            elif self.rvs_mode == "RvS-RC":
                obs_next = np.append(obs_next, [cur_ret, cur_cost])
            elif self.rvs_mode == "RvS-RCR":
                if "Circle" in task:
                    states_xa = np.append(states_xa, obs_next[0])
                    xa = torch.tensor(states_xa, dtype=torch.float).reshape([1, states_xa.shape[0], 1])
                    cost_inputs = ((xa, xa), ((xa, xa)))
                if "Run" in task:
                    xdot = 2; ydot = 3
                    if "Drone" in task or "Ant" in task:
                        xdot += 1; ydot += 1
                    states_ya = np.append(states_ya, obs_next[1])
                    states_va = np.append(states_va, np.hypot(obs_next[xdot], obs_next[ydot]))
                    # print(f"y = {obs_next[1]}, v = {np.hypot(obs_next[xdot], obs_next[ydot])}")
                    ya = torch.tensor(states_ya, dtype=torch.float).reshape([1, states_ya.shape[0], 1])
                    va = torch.tensor(states_va, dtype=torch.float).reshape([1, states_va.shape[0], 1])
                    cost_inputs = ((ya, ya), (va, va))
                    
                cost_rob = eval_specification.robustness(cost_inputs)[0, 0, 0]
                obs_next = np.append(obs_next, [cur_ret, target_cost, cost_rob])

            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost, rew_rob, cost_rob
