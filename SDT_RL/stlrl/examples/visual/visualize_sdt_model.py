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

import yaml
import argparse

def load_reqs(model_dir: str, best_model):

    env = gym.make(config_dict['task'])
    env.reset()

    env.set_target_cost(config_dict['cost_limit'])
    env = wrap_env(env, reward_scale=config_dict['reward_scale'])
    env = OfflineEnvWrapper(env)

    model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=config_dict['embedding_dim'],
        seq_len=config_dict['seq_len'],
        episode_len=config_dict['episode_len'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict['num_heads'],
        attention_dropout=config_dict['attention_dropout'],
        residual_dropout=config_dict['residual_dropout'],
        embedding_dropout=config_dict['embedding_dropout'],
        time_emb=config_dict['time_emb'],
        use_rew=config_dict['use_rew'],
        use_cost=config_dict['use_cost'],
        use_rew_prefix=config_dict['use_rew_prefix'],
        use_rew_suffix=config_dict['use_rew_suffix'],
        use_cost_prefix=config_dict['use_cost_prefix'],
        use_cost_suffix=config_dict['use_cost_suffix'],
        cost_transform=config_dict['cost_transform'],
        add_cost_feat=config_dict['add_cost_feat'],
        mul_cost_feat=config_dict['mul_cost_feat'],
        cat_cost_feat=config_dict['cat_cost_feat'],
        action_head_layers=config_dict['action_head_layers'],
        cost_prefix=config_dict['cost_prefix'],
        stochastic=config_dict['stochastic'],
        init_temperature=config_dict['init_temperature'],
        target_entropy=-env.action_space.shape[0],
    ).to(config_dict['device'])

    checkpoint = torch.load(model_file, map_location=config_dict['device'])

    # Handle both raw state_dict and wrapped checkpoint
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, env


@torch.no_grad()
def run_episode(model, env, args, target_reward: float, target_cost: float):
    """
    Run one episode with a rolling context window.
    CDT is a Decision Transformer — it conditions on the full
    (state, action, return-to-go, cost-to-go) sequence so far.
    """
    seq_len = args.seq_len
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = args.device

    # Rolling buffers (oldest entries will be sliced off)
    states = torch.zeros((1, seq_len, state_dim), device=device)
    actions = torch.zeros((1, seq_len, action_dim), device=device)
    returns = torch.zeros((1, seq_len, 1), device=device)
    costs_return = torch.zeros((1, seq_len, 1), device=device)
    time_steps = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    mask = torch.zeros((1, seq_len), device=device)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # Scale targets the same way your trainer does
    target_rew = target_reward * args.reward_scale
    target_cost_val = target_cost * args.cost_scale

    episode_reward, episode_cost, episode_length = 0.0, 0.0, 0

    for t in range(args.episode_len):
        # Shift window and insert current step at the last position
        pos = min(t, seq_len - 1)

        states[0, pos] = obs
        returns[0, pos, 0] = target_rew
        costs_return[0, pos, 0] = target_cost_val
        time_steps[0, pos] = t
        mask[0, pos] = 1.0

        # Use only the filled portion (or full window once warm)
        start = max(0, t + 1 - seq_len)
        s = states[:, start:pos + 1]
        a = actions[:, start:pos + 1]
        r = returns[:, start:pos + 1]
        c = costs_return[:, start:pos + 1]
        ts = time_steps[:, start:pos + 1]
        m = mask[:, start:pos + 1]

        # CDT forward — adjust argument names to match your CDT.get_action / forward API
        action = model.get_action(s, a, r, c, ts, m)

        if isinstance(action, tuple):
            action = action[0]  # stochastic returns (mean, log_std)

        action_np = action.squeeze().cpu().numpy()
        action_np = np.clip(action_np, -env.action_space.high[0], env.action_space.high[0])

        # Store action taken
        actions[0, pos] = action

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        cost = info.get("cost", 0.0)

        episode_reward += reward
        episode_cost += cost
        episode_length += 1

        # Decay return targets (standard Decision Transformer receding horizon)
        target_rew = max(target_rew - reward, 0)
        target_cost_val = max(target_cost_val - cost, 0)

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        env.render()

        if terminated or truncated:
            break

    return episode_reward, episode_cost, episode_length


def visualize(model_dir: str, best_model):

    config_yaml = f"{model_dir}/../config.yaml"

    if best_model:
        model_file = f"{model_dir}/model_best.pt"
    else:
        model_file = f"{model_dir}/model.pt"

    with open(config_yaml, 'r') as stream:
        try:
            # Converts yaml document to python object
            config_dict=yaml.full_load(stream)
            # Printing dictionary
            # print(config_dict)
        except yaml.YAMLError as e:
            print(e)
            return
    
    # initialize environment
    if "Metadrive" in config_dict['task']:
        import gym
    else:
        import gymnasium as gym  # noqa
    
    # --- Load config the same way train() does ---
    # You can also just point to your saved config.yaml via pyrallis
    args_raw = CDTTrainConfig()
    cfg = asdict(CDT_DEFAULT_CONFIG[args_raw.task]())
    args = types.SimpleNamespace(**cfg)

    # ✏️  Override these to match your run
    args.task = "OfflineCarCircle-v0"
    args.cost_limit = 10
    args.device = "cpu"

    checkpoint_path = model_file
    target_reward =    # match the target_returns you used in eval
    target_cost = 10

    model, env = load_reqs(checkpoint_path, args)

    num_episodes = 5
    for ep in range(num_episodes):
        rew, cost, length = run_episode(
            model, env, args,
            target_reward=target_reward,
            target_cost=target_cost,
        )
        print(f"Episode {ep+1:02d} | Reward: {rew:.1f} | Cost: {cost:.1f} | Length: {length}")

    env.close()

def visualize_old(model_dir, best_model):
    config_yaml = f"{model_dir}/../config.yaml"

    if best_model:
        model = f"{model_dir}/model_best.pt"
    else:
        model = f"{model_dir}/model.pt"

    with open(config_yaml, 'r') as stream:
        try:
            # Converts yaml document to python object
            config_dict=yaml.full_load(stream)
            # Printing dictionary
            # print(config_dict)
        except yaml.YAMLError as e:
            print(e)
            return
    
    # initialize environment
    if "Metadrive" in config_dict['task']:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = gym.make(config_dict['task'])
    env.reset()

    # # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(config_dict['cost_limit'])

        # model & optimizer & scheduler setup
    model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=config_dict['embedding_dim'],
        seq_len=config_dict['seq_len'],
        episode_len=config_dict['episode_len'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict['num_heads'],
        attention_dropout=config_dict['attention_dropout'],
        residual_dropout=config_dict['residual_dropout'],
        embedding_dropout=config_dict['embedding_dropout'],
        time_emb=config_dict['time_emb'],
        use_rew=config_dict['use_rew'],
        use_cost=config_dict['use_cost'],
        use_rew_prefix=config_dict['use_rew_prefix'],
        use_rew_suffix=config_dict['use_rew_suffix'],
        use_cost_prefix=config_dict['use_cost_prefix'],
        use_cost_suffix=config_dict['use_cost_suffix'],
        cost_transform=config_dict['cost_transform'],
        add_cost_feat=config_dict['add_cost_feat'],
        mul_cost_feat=config_dict['mul_cost_feat'],
        cat_cost_feat=config_dict['cat_cost_feat'],
        action_head_layers=config_dict['action_head_layers'],
        cost_prefix=config_dict['cost_prefix'],
        stochastic=config_dict['stochastic'],
        init_temperature=config_dict['init_temperature'],
        target_entropy=-env.action_space.shape[0],
    ).to(config_dict['device'])
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}
    
    # trainer
    trainer = CDTTrainer(model,
                         env,
                         logger=logger,
                         learning_rate=config_dict['learning_rate'],
                         weight_decay=config_dict['weight_decay'],
                         betas=config_dict['betas'],
                         clip_grad=config_dict['clip_grad'],
                         lr_warmup_steps=config_dict['lr_warmup_steps'],
                         reward_scale=config_dict['reward_scale'],
                         cost_scale=config_dict['cost_scale'],
                         rob_scale=config_dict['rob_scale'],
                         loss_cost_weight=config_dict['loss_cost_weight'],
                         loss_state_weight=config_dict['loss_state_weight'],
                         loss_spec_weight=config_dict['loss_spec_weight'],
                         cost_reverse=config_dict['cost_reverse'],
                         no_entropy=config_dict['no_entropy'],
                         device=config_dict['device'])
    
    # similar to CDT: lower cost trajs have higher probabilities
    # for SDT, higher cost rob trajs have higher probabilities
    cost_transform = lambda x: 100 - x if config_dict['linear'] else 1 / (x + 10)
    if use_cost_rob:
        if "Circle" in config_dict['task']:
            cost_transform = lambda x: x + 0.15
        if "Run" in config_dict['task']:
            cost_transform = lambda x: x + 0.3

    dataset = SequenceDataset(
        data,
        seq_len=config_dict['seq_len'],
        reward_scale=config_dict['reward_scale'],
        cost_scale=config_dict['cost_scale'],
        deg=config_dict['deg'],
        pf_sample=config_dict['pf_sample'],
        max_rew_decrease=config_dict['max_rew_decrease'],
        beta=config_dict['beta'],
        augment_percent=config_dict['augment_percent'],
        cost_reverse=config_dict['cost_reverse'],
        max_reward=config_dict['max_reward'],
        min_reward=config_dict['min_reward'],
        pf_only=config_dict['pf_only'],
        rmin=config_dict['rmin'],
        cost_bins=config_dict['cost_bins'],
        npb=config_dict['npb'],
        cost_sample=config_dict['cost_sample'],
        cost_transform=cost_transform,
        start_sampling=config_dict['start_sampling'],
        prob=config_dict['prob'],
        random_aug=config_dict['random_aug'],
        aug_rmin=config_dict['aug_rmin'],
        aug_rmax=config_dict['aug_rmax'],
        aug_cmin=config_dict['aug_cmin'],
        aug_cmax=config_dict['aug_cmax'],
        cgap=config_dict['cgap'],
        rstd=config_dict['rstd'],
        cstd=config_dict['cstd'],
        specification=specification,
        rob_scale=config_dict['rob_scale'],
        use_rew_prefix=config_dict['use_rew_prefix'],
        use_rew_suffix=config_dict['use_rew_suffix'],
        use_cost_prefix=config_dict['use_cost_prefix'],
        use_cost_suffix=config_dict['use_cost_suffix'],
        task=config_dict['task']
    )

    def to_device(data):
        if isinstance(data, list):
            return [d.to(config_dict['device']) for d in data]
        else:
            return data.to(config_dict['device'])
        
    for step in trange(config_dict['update_steps'], desc="Training"):
        batch = next(trainloader_iter)
        
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            to_device(b) for b in batch
        ]
        # print(returns.shape, costs_return.shape)
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               episode_cost, costs)
        if config_dict['render']:
            env.render()

        # evaluation
        if (step + 1) % config_dosc['eval_every'] == 0 or step == config_disc['update_steps'] - 1:
            average_reward, average_cost, average_safe_rate = [], [], []
            log_cost, log_reward, log_len, log_safe_rate = {}, {}, {}, {}
            for target_return in config_dict['target_returns']:
                reward_return, cost_return = target_return
                ret, cost, length, rew_rob, cost_rob, safe_rate = trainer.evaluate(
                        config_dict['eval_episodes'], 
                        reward_return * config_dict['reward_scale'],
                        cost_return * config_dict['cost_scale'],
                        specification,
                        config_dict['use_rew_prefix'], 
                        config_dict['use_cost_prefix'],
                        config_dict['task'],
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


def train(args: CDTTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(CDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CDT_DEFAULT_CONFIG[config_dict['task']]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # set seed
    seed_all(config_dict['seed'])
    if config_dict['device == "cpu"']:
        torch.set_num_threads(config_dict['threads'])

    # initialize environment
    if "Metadrive" in config_dict['task']:
        import gym
    else:
        import gymnasium as gym  # noqa
    env = gym.make(config_dict['task'])
    env.reset()

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(config_dict['cost_limit'])

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if config_dict['density != 1.0']:
        density_cfg = DENSITY_CFG[config_dict['task'] + "_density" + str(config_dict['density'])]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                config_dict['outliers_percent'],
                                config_dict['noise_scale'],
                                config_dict['inpaint_ranges'],
                                config_dict['epsilon'],
                                config_dict['density'],
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=config_dict['reward_scale'],
    )
    env = OfflineEnvWrapper(env)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization code for SDT created models')
    parser.add_argument('--model_dir',              default="INVALID_MODEL",     type=str,    help='model to choose', metavar='')
    parser.add_argument('--best_model',              default=False,     type=bool,    help='whether to choose best model', metavar='')

    ARGS = parser.parse_args()

    visualize(**vars(ARGS))