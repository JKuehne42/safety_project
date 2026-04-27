from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import RvS, RvSTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from stlcg import LessThan, GreaterThan, And, Or, Always, Implies, Negation, Eventually


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    returns: List[float] = field(default=[350, 400, 450], is_mutable=True)
    costs: List[float] = field(default=[10, 10, 10], is_mutable=True)
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = False
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):

    cfg, model = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )

    # model & optimizer & scheduler setup
    state_dim = env.observation_space.shape[0]
    state_dim = env.observation_space.shape[0]
    if cfg["prefix"] == "RvS-R":
        state_dim += 1
    elif cfg["prefix"] == "RvS-RC":
        state_dim += 2
    elif cfg["prefix"] == "RvS-RCR":
        state_dim += 3
    else:
        raise NotImplementedError
    
    rvs_model = RvS(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    rvs_model.load_state_dict(model["model_state"])
    rvs_model.to(args.device)

    trainer = RvSTrainer(rvs_model,
                         env,
                         rvs_mode=cfg["prefix"],
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         device=args.device)

    with torch.no_grad():
        if "Circle" in cfg["task"]:
            threshold = 5
            x_lim = 0.1 * 6  # scalar * x_lim
            ϕ_xa = And(LessThan(lhs='xa', val=x_lim), GreaterThan(lhs='xa', val=-x_lim))
            specification = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, interval=[1, threshold])))
            
        if "Run" in cfg["task"]:
            threshold = 5
            y_lim = 0.1 * 2
            ϕ_ya = And(LessThan(lhs='ya', val=y_lim), GreaterThan(lhs='ya', val=-y_lim))
            v_lim = {"OfflineAntRun-v0": 0.45, "OfflineBallRun-v0": 0.5, "OfflineCarRun-v0": 1.5, "OfflineDroneRun-v0": 0.3}
            ϕ_va = LessThan(lhs="va", val=v_lim[cfg["task"]])
            specification = Always(And(φ_ya, Implies(Negation(φ_va), Eventually(φ_va, [1, threshold]))))

    for target_reward, target_cost in zip(args.returns, args.costs):
        seed_all(cfg["seed"])
        ret, cost, length, rew_rob, cost_rob, safe_rate = trainer.evaluate(args.eval_episodes, 
                                             target_reward*cfg["reward_scale"],
                                             target_cost*cfg["cost_scale"],
                                             specification, cfg["task"], threshold)
        print(
            f"Eval safe rate = {safe_rate}, Target reward {target_reward}, reward {ret}; Target cost = {target_cost}, cost {cost}"
        )


if __name__ == "__main__":
    eval()
