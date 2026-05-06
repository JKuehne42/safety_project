from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa
from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp
from stlcg.stlcg import STL_Formula


class CDT(nn.Module):
    """
    Constrained Decision Transformer (CDT)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        seq_len (int): The length of the sequence to process.
        episode_len (int): The length of the episode.
        embedding_dim (int): The dimension of the embeddings.
        num_layers (int): The number of transformer layers to use.
        num_heads (int): The number of heads to use in the multi-head attention.
        attention_dropout (float): The dropout probability for attention layers.
        residual_dropout (float): The dropout probability for residual layers.
        embedding_dropout (float): The dropout probability for embedding layers.
        time_emb (bool): Whether to include time embeddings.
        use_rew (bool): Whether to include return embeddings.
        use_cost (bool): Whether to include cost embeddings.
        cost_transform (bool): Whether to transform the cost values.
        add_cost_feat (bool): Whether to add cost features.
        mul_cost_feat (bool): Whether to multiply cost features.
        cat_cost_feat (bool): Whether to concatenate cost features.
        action_head_layers (int): The number of layers in the action head.
        cost_prefix (bool): Whether to include a cost prefix.
        stochastic (bool): Whether to use stochastic actions.
        init_temperature (float): The initial temperature value for stochastic actions.
        target_entropy (float): The target entropy value for stochastic actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        use_rew_prefix: bool = False,
        use_rew_suffix: bool = False,
        use_cost_prefix: bool = False,
        use_cost_suffix: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        self.use_rew_prefix = use_rew_prefix
        self.use_rew_suffix = use_rew_suffix
        self.use_cost_prefix = use_cost_prefix
        self.use_cost_suffix = use_cost_suffix
        if self.use_cost or self.use_cost_suffix:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew or self.use_rew_suffix:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew_prefix:
            self.rew_rob_prefix_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_cost_prefix:
            self.cost_rob_prefix_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=dt_seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        # a classification problem
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        if self.stochastic:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.apply(self._init_weights)

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            rew_rob_prefix: torch.Tensor,  # [batch_size, seq_len]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len], RTG or rew_rob_suffix
            cost_rob_prefix: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len], CTG or cost_rob_suffix
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            episode_cost: torch.Tensor = None,  # [batch_size, ]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]
        
        if self.cost_transform is not None and self.use_cost:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost_suffix or self.use_cost_prefix:
            # SDT: prefix, suffix, R, S, A
            if self.use_rew:
                returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, returns_emb)
            if self.use_rew_prefix:
                rew_rob_prefix_emb = self.rew_rob_prefix_emb(rew_rob_prefix.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, rew_rob_prefix_emb)
            if self.use_cost_suffix:
                costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, costs_emb)
            if self.use_cost_prefix:
                cost_rob_prefix_emb = self.cost_rob_prefix_emb(cost_rob_prefix.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, cost_rob_prefix_emb)
        else:
            # CDT: R, C, S, A
            if self.use_cost:
                costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, costs_emb)
            if self.use_rew:
                returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
                seq_list.insert(0, returns_emb)

        # [batch_size, seq_len, 2-4, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len,
                                    self.embedding_dim)

        if padding_mask is not None:
            # [batch_size, seq_len * self.seq_repeat], stack mask identically to fit the sequence
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len * self.seq_repeat, embedding_dim]
        out = self.out_norm(out)

        # [batch_size, seq_len, self.seq_repeat, embedding_dim]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        # [batch_size, self.seq_repeat, seq_len, embedding_dim]
        out = out.permute(0, 2, 1, 3)

        # [batch_size, seq_len, embedding_dim]
        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            # cost_prefix feature, deprecated
            # [batch_size, seq_len, 2 * embedding_dim]
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        # get predictions

        action_preds = self.action_head(
            state_feat
        )  # predict next action given state, [batch_size, seq_len, action_dim]
        # [batch_size, seq_len, 2]
        cost_preds = self.cost_pred_head(
            action_feature)  # predict next cost return given state and action
        cost_preds = F.log_softmax(cost_preds, dim=-1)

        # [batch_size, seq_len, obs_dim]
        state_preds = self.state_pred_head(
            action_feature)  # predict next state given state and action

        return action_preds, cost_preds, state_preds


class CDTTrainer:
    """
    Constrained Decision Transformer Trainer
    
    Args:
        model (CDT): A CDT model to train.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        betas (Tuple[float, ...]): The betas for the optimizer.
        clip_grad (float): The clip gradient value.
        lr_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        loss_cost_weight (float): The weight for the cost loss.
        loss_state_weight (float): The weight for the state loss.
        cost_reverse (bool): Whether to reverse the cost.
        no_entropy (bool): Whether to use entropy.
        device (str): The device to use for training (e.g. "cpu" or "cuda").

    """

    def __init__(
            self,
            model: CDT,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            rob_scale: float = -1,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            loss_spec_weight: float = 0.0,
            cost_reverse: bool = False,
            no_entropy: bool = False,
            device="cpu") -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.rob_scale = rob_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.spec_weight = loss_spec_weight
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.stochastic = self.model.stochastic
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        self.max_action = self.model.max_action

        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask,
                       episode_cost, costs):
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        rew_rob_prefix, cost_rob_prefix = None, None
        if isinstance(returns, list):
            rew_rob_prefix, returns = returns
        if isinstance(costs_return, list):
            cost_rob_prefix, costs_return = costs_return
        # print(states.dtype, actions.dtype, rew_rob_prefix.dtype, returns.dtype, cost_rob_prefix.dtype, costs_return.dtype, time_steps.dtype)
        action_preds, cost_preds, state_preds = self.model(
            states=states,
            actions=actions,
            rew_rob_prefix=rew_rob_prefix,
            returns_to_go=returns,
            cost_rob_prefix=cost_rob_prefix,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
            episode_cost=episode_cost,
        )

        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            # log_likelihood = action_preds.log_prob(actions).sum(axis=-1)[mask > 0].mean()
            # entropy = action_preds.entropy().sum(axis=-1)[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            entropy_reg_item = entropy_reg.item()
            if self.no_entropy:
                entropy_reg = 0.0
                entropy_reg_item = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            self.logger.store(tab="train",
                              nll=-log_likelihood.item(),
                              ent=entropy.item(),
                              ent_reg=entropy_reg_item)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        # cost_preds: [batch_size * seq_len, 2], costs: [batch_size * seq_len]
        cost_preds = cost_preds.reshape(-1, 2)
        costs = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
        # cost_loss = F.mse_loss(cost_preds, costs.detach(), reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()
        # compute the accuracy, 0 value, 1 indice, [batch_size, seq_len]
        pred = cost_preds.data.max(dim=1)[1]
        correct = pred.eq(costs.data.view_as(pred)) * mask.flatten()
        correct = correct.sum()
        total_num = mask.sum()
        acc = correct / total_num

        # [batch_size, seq_len, state_dim]
        state_loss = F.mse_loss(state_preds[:, :-1],
                                states[:, 1:].detach(),
                                reduction="none")
        state_loss = (state_loss * mask[:, 1:].unsqueeze(-1)).mean()

        spec_loss = 0
        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss + self.spec_weight * spec_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() *
                                (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        self.scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            act_loss=act_loss.item(),
            cost_loss=cost_loss.item(),
            cost_acc=acc.item(),
            state_loss=state_loss.item(),
            train_lr=self.scheduler.get_last_lr()[0],
        )

    def evaluate(self, num_rollouts, target_return, target_cost,
                 eval_specification: Tuple[STL_Formula, STL_Formula] = (None, None),
                 use_rew_prefix: bool = False,
                 use_cost_prefix: bool = False,
                 task: str = None,
                 threshold: int = 5):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        
        episode_rets, episode_costs, episode_lens = [], [], []
        episode_rew_rob, episode_cost_rob = [], []
        ϕ_reward, ϕ_cost = eval_specification
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            ret, len, cost, rew_rob, cost_rob = self.rollout(
                self.model, self.env, target_return, target_cost,
                eval_specification, use_rew_prefix, use_cost_prefix,
                task, threshold)
            episode_rets.append(ret); episode_costs.append(cost); episode_lens.append(len)
            episode_rew_rob.append(rew_rob); episode_cost_rob.append(cost_rob)

        episode_rew_rob = None if ϕ_reward is None else np.array(episode_rew_rob)
        episode_cost_rob = None if ϕ_cost is None else np.array(episode_cost_rob)
        safe_rate = np.sum(np.array(episode_costs) <= 0.0) / num_rollouts

        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens), episode_rew_rob, episode_cost_rob, safe_rate

    @torch.no_grad()
    def rollout(
        self,
        model: CDT,
        env: gym.Env,
        target_return: float, # target_reward / target_reward_rob
        target_cost: float, # target_cost / target_cost_rob
        eval_specification: Tuple[STL_Formula, STL_Formula] = (None, None),
        use_rew_prefix: bool = False,
        use_cost_prefix: bool = False,
        task: str = None,
        threshold: int = 5
    ) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(1, model.episode_len + 1, model.state_dim,
                             dtype=torch.float, device=self.device)
        actions = torch.zeros(1, model.episode_len, model.action_dim,
                              dtype=torch.float, device=self.device)
        returns = torch.zeros(1, model.episode_len + 1,
                              dtype=torch.float, device=self.device)
        rew_rob_prefix = torch.zeros(1, model.episode_len + 1, 
                                     dtype=torch.float, device=self.device)
        costs = torch.zeros(1, model.episode_len + 1,
                            dtype=torch.float, device=self.device)
        cost_rob_prefix = torch.zeros(1, model.episode_len + 1,
                                      dtype=torch.float, device=self.device)
        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long, device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)
        
        epi_cost = torch.tensor(np.array([target_cost]),
                                dtype=torch.float,
                                device=self.device)   
 
        ϕ_reward, ϕ_cost = eval_specification

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        consecutive_ones = 0
        # for step in trange(model.episode_len, desc="Rollout ...", leave=False):
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            # compute robustness value prefix
            # [step, obs_dim]
            eval_obs = states[:, :step + 1].squeeze(dim=0)

            if "Circle" in task:
                xa = eval_obs[:, 0]
                xa = xa.reshape([1, xa.shape[0], 1])
            if "Run" in task:
                xdot = 2; ydot = 3
                if "Drone" in task or "Ant" in task:
                    xdot += 1; ydot += 1
                ya = eval_obs[:, 1]
                va = torch.hypot(eval_obs[:, xdot], eval_obs[:, ydot])
                ya = ya.reshape([1, ya.shape[0], 1])
                va = va.reshape([1, va.shape[0], 1])

            if "Jump" in task:
                za = eval_obs[:, 0]
                ya = eval_obs[:, 1]
                xa = eval_obs[:, 2]
                za = za.reshape([1, za.shape[0], 1])
                ya = ya.reshape([1, ya.shape[0], 1])
                xa = xa.reshape([1, xa.shape[0], 1])

            if use_rew_prefix:
                pass
            if use_cost_prefix:
                if "Circle" in task:
                    cost_inputs = ((xa, xa), ((xa, xa)))
                if "Run" in task:
                    cost_inputs = ((ya, ya), (va, va))
                if "Jump" in task:
                    cost_inputs = (za, ((ya, ya), (xa, xa)))
                cost_rob = ϕ_cost.robustness(cost_inputs, scale=self.rob_scale)
                cost_rob_prefix[:, step] = cost_rob[0, 0, 0]
                # cost_rob = ϕ_cost.forward(cost_inputs, scale=self.rob_scale)
                # cost_rob_prefix[:, :step+1] = cost_rob[0, :, 0]
            
            rrp = rew_rob_prefix[:, :step + 1][:, -model.seq_len:]
            crp = cost_rob_prefix[:, :step + 1][:, -model.seq_len:]
            
            acts, _, _ = model(s, a, rrp, r, crp, c, t, None, epi_cost)
            if self.stochastic:
                acts = acts.mean
            acts = acts.clamp(-self.max_action, self.max_action)
            act = acts[0, -1].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            
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
            if "Jump" in task:
                # Jump task: check height and boundary constraints
                cost_height = info.get("cost_height_violation", 0.0)
                cost_bounds = info.get("cost_outside_bounds", 0.0)
                cost = np.min([1, cost_height + cost_bounds]) * self.cost_scale
            
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward) if ϕ_reward is None \
                else torch.as_tensor(target_return)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost) if ϕ_cost is None \
                else torch.as_tensor(target_cost)
            
            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += cost

            if terminated or truncated:
                break

        rew_rob, cost_rob = None, None
        cost_rob = episode_cost
        if ϕ_reward is not None or ϕ_cost is not None:
            if self.device == "cpu":
                cost_rob = cost_rob_prefix[0, model.episode_len-1].numpy()
            else:
                cost_rob = cost_rob_prefix[0, model.episode_len-1].cpu().numpy()

        return episode_ret, episode_len, episode_cost, rew_rob, cost_rob

    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)

        acts, _, _ = model(s, a, r, c, t, None, epi_cost)
        if self.stochastic:
            acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        acts = acts.clamp(-self.max_action, self.max_action)
        act = acts[0, -1].cpu().numpy()
        return act

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
