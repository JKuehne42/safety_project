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
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.rvs_configs import RvS_DEFAULT_CONFIG, RvSTrainConfig
from osrl.algorithms import RvS, RvSTrainer
from osrl.common import TransitionDataset
from osrl.common.dataset import process_rvs_dataset
from osrl.common.exp_util import auto_name, seed_all
from stlcg import LessThan, GreaterThan, And, Or, Always, Implies, Negation, Eventually


@pyrallis.wrap()
def train(args: RvSTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(RvSTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(RvS_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(RvS_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # the cost scale is down in trainer rollout
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym  # noqa
    env = gym.make(args.task)
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

    with torch.no_grad():
        if "Circle" in args.task:
            threshold = 5
            x_lim = 0.1 * 6  # scalar * x_lim
            ϕ_xa = And(LessThan(lhs='xa', val=x_lim), GreaterThan(lhs='xa', val=-x_lim))
            spec = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, interval=[1, threshold])))
            
        if "Run" in args.task:
            y_lim = 0.1 * 2
            ϕ_ya = And(LessThan(lhs='ya', val=y_lim), GreaterThan(lhs='ya', val=-y_lim))
            v_lim = {"OfflineAntRun-v0": 0.45, "OfflineBallRun-v0": 0.5, "OfflineCarRun-v0": 1.5, "OfflineDroneRun-v0": 0.3}
            ϕ_va = LessThan(lhs="va", val=v_lim[args.task])
            spec = Always(And(φ_ya, Implies(Negation(φ_va), Eventually(φ_va, [1, threshold]))))

    process_rvs_dataset(data, args.reward_scale, args.cost_scale, 
                        args.gamma, True, spec, args.prefix, args.task)

    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # model & optimizer & scheduler setup
    state_dim = env.observation_space.shape[0]
    if args.prefix == "RvS-R":
        state_dim += 1
    elif args.prefix == "RvS-RC":
        state_dim += 2
    elif args.prefix == "RvS-RCR":
        state_dim += 3
    else:
        raise NotImplementedError

    model = RvS(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = RvSTrainer(model,
                        env,
                        logger=logger,
                        actor_lr=args.actor_lr,
                        rvs_mode=args.prefix,
                        cost_limit=args.cost_limit,
                        reward_scale=args.reward_scale,
                        cost_scale=args.cost_scale,
                        # relabel_cost=args.relabel_cost,
                        device=args.device)

    trainloader = DataLoader(
        TransitionDataset(data),
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

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, _, actions, _, _, _ = [b.to(args.device) for b in batch]
        trainer.train_one_step(observations, actions)

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
                            spec, args.task, threshold)
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
            # if cost < best_cost or (cost == best_cost and ret > best_reward):
            if safe_rate > best_safe_rate or (safe_rate == best_safe_rate
                                              and ret > best_reward):
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
