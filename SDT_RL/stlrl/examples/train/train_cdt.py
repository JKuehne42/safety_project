import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger, TensorboardLogger, DummyLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
from osrl.algorithms import CDT, CDTTrainer
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all

from stlcg import LessThan, GreaterThan, And, Or, Always, Implies, Negation, Eventually
from stlpy.STL import LinearPredicate

@pyrallis.wrap()
def train(args: CDTTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(CDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CDT_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(CDT_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    # logger = DummyLogger()
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym  # noqa
    env = gym.make(args.task)
    env.reset()
    # env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)
    
    # STL specification
    # pscale = 1; scale = -1
    with torch.no_grad():
        
        if "Circle" in args.task:
            threshold = 5
            x_lim = 0.1 * 6  # scalar * x_lim
            ϕ_xa = And(LessThan(lhs='xa', val=x_lim), GreaterThan(lhs='xa', val=-x_lim))

            # ϕ_cost = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, interval=[1, 5])))
            ϕ_cost = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, [1, threshold])))
            ϕ_reward = None
            
        if "Run" in args.task:
            threshold = 5
            y_lim = 0.1 * 2
            ϕ_ya = And(LessThan(lhs='ya', val=y_lim), GreaterThan(lhs='ya', val=-y_lim))
            v_lim = {"OfflineAntRun-v0": 0.45, "OfflineBallRun-v0": 0.5, "OfflineCarRun-v0": 1.5, "OfflineDroneRun-v0": 0.3}
            ϕ_va = LessThan(lhs="va", val=v_lim[args.task])
            # And(Eventually(φ_va, [1, 5]), Always(φ_va, [1, 10]))
            
            ϕ_cost = Always(And(φ_ya, Implies(Negation(φ_va), Eventually(φ_va, [1, threshold]))))
            ϕ_reward = None

    use_rew_rob = args.use_rew_suffix or args.use_rew_prefix
    use_cost_rob = args.use_cost_suffix or args.use_cost_prefix
    if use_rew_rob and use_cost_rob:
        specification = (ϕ_reward, ϕ_cost)
    elif use_rew_rob and not use_cost_rob:
        specification = (ϕ_reward, None)
    elif not use_rew_rob and use_cost_rob:
        specification = (None, ϕ_cost)
    else:
        specification = (None, None)
    print(f"specification = {specification}")

    # model & optimizer & scheduler setup
    model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        time_emb=args.time_emb,
        use_rew=args.use_rew,
        use_cost=args.use_cost,
        use_rew_prefix=args.use_rew_prefix,
        use_rew_suffix=args.use_rew_suffix,
        use_cost_prefix=args.use_cost_prefix,
        use_cost_suffix=args.use_cost_suffix,
        cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
    ).to(args.device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = CDTTrainer(model,
                         env,
                         logger=logger,
                         learning_rate=args.learning_rate,
                         weight_decay=args.weight_decay,
                         betas=args.betas,
                         clip_grad=args.clip_grad,
                         lr_warmup_steps=args.lr_warmup_steps,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         rob_scale=args.rob_scale,
                         loss_cost_weight=args.loss_cost_weight,
                         loss_state_weight=args.loss_state_weight,
                         loss_spec_weight=args.loss_spec_weight,
                         cost_reverse=args.cost_reverse,
                         no_entropy=args.no_entropy,
                         device=args.device)

    # similar to CDT: lower cost trajs have higher probabilities
    # for SDT, higher cost rob trajs have higher probabilities
    cost_transform = lambda x: 100 - x if args.linear else 1 / (x + 10)
    if use_cost_rob:
        if "Circle" in args.task:
            cost_transform = lambda x: x + 0.15
        if "Run" in args.task:
            cost_transform = lambda x: x + 0.3

    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta,
        augment_percent=args.augment_percent,
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only,
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=cost_transform,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
        specification=specification,
        rob_scale=args.rob_scale,
        use_rew_prefix=args.use_rew_prefix,
        use_rew_suffix=args.use_rew_suffix,
        use_cost_prefix=args.use_cost_prefix,
        use_cost_suffix=args.use_cost_suffix,
        task=args.task
    )

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_safe_rate = -np.inf
    best_idx = 0

    def to_device(data):
        if isinstance(data, list):
            return [d.to(args.device) for d in data]
        else:
            return data.to(args.device)

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            to_device(b) for b in batch
        ]
        # print(returns.shape, costs_return.shape)
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               episode_cost, costs)
        if args.render:
            env.render()

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost, average_safe_rate = [], [], []
            log_cost, log_reward, log_len, log_safe_rate = {}, {}, {}, {}
            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                ret, cost, length, rew_rob, cost_rob, safe_rate = trainer.evaluate(
                        args.eval_episodes, 
                        reward_return * args.reward_scale,
                        cost_return * args.cost_scale,
                        specification,
                        args.use_rew_prefix, 
                        args.use_cost_prefix,
                        args.task,
                        threshold)
                average_cost.append(cost)
                average_reward.append(ret)
                average_safe_rate.append(safe_rate)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_return))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})
                log_safe_rate.update({name: safe_rate})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)
            logger.store(tab="safe_rate", **log_safe_rate)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)
            mean_safe_rate = np.mean(average_safe_rate)
            # if mean_cost < best_cost or (mean_cost == best_cost
            #                              and mean_ret > best_reward):
            if mean_safe_rate > best_safe_rate or (mean_safe_rate == best_safe_rate
                                                   and mean_ret > best_reward):
            # if (mean_safe_rate > best_safe_rate and mean_cost < best_cost) or \
            #     (mean_safe_rate == best_safe_rate and mean_ret > best_reward):
                best_safe_rate = mean_safe_rate
                best_cost = mean_cost
                best_reward = mean_ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
            

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
