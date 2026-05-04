from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import CDT, CDTTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from stlcg import LessThan, GreaterThan, And, Always, Implies, Negation, Eventually


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt" # Set this to the parent folder to the checkpoint folder
    returns: List[float] = field(default=[350, 400, 450], is_mutable=True)
    costs: List[float] = field(default=[10, 10, 10], is_mutable=True)
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = False
    max: bool = True
    device: str = "cpu"
    threads: int = 4
    render_mode: str = "human"


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
        # env=gym.make(cfg["task"], render_mode=args.render_mode),
        env=gym.make(cfg['task']),
        reward_scale=cfg["reward_scale"],
    )
    
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])
    env.reset()


    # model & optimizer & scheduler setup
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=cfg["use_rew"],
        use_cost=cfg["use_cost"],
        use_rew_prefix=cfg["use_rew_prefix"],
        use_rew_suffix=cfg["use_rew_suffix"],
        use_cost_prefix=cfg["use_cost_prefix"],
        use_cost_suffix=cfg["use_cost_suffix"],
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=-env.action_space.shape[0],
    )
    cdt_model.load_state_dict(model["model_state"])
    cdt_model.to(args.device)

    trainer = CDTTrainer(cdt_model,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         cost_reverse=cfg["cost_reverse"],
                         rob_scale=cfg["rob_scale"],
                         device=args.device)

    # STL specification
    with torch.no_grad():
        
        if "Circle" in cfg["task"]:
            threshold = 5
            x_lim = 0.1 * 6  # scalar * x_lim
            ϕ_xa = And(LessThan(lhs='xa', val=x_lim), GreaterThan(lhs='xa', val=-x_lim))

            # ϕ_cost = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, interval=[1, 5])))
            ϕ_cost = Always(Implies(Negation(ϕ_xa), Eventually(ϕ_xa, [1, threshold])))
            ϕ_reward = None
            
        if "Run" in cfg["task"]:
            threshold = 5
            y_lim = 0.1 * 2
            ϕ_ya = And(LessThan(lhs='ya', val=y_lim), GreaterThan(lhs='ya', val=-y_lim))
            v_lim = {"OfflineAntRun-v0": 0.45, "OfflineBallRun-v0": 0.5, "OfflineCarRun-v0": 1.5, "OfflineDroneRun-v0": 0.3}
            ϕ_va = LessThan(lhs="va", val=v_lim[cfg["task"]])
            # And(Eventually(φ_va, [1, 5]), Always(φ_va, [1, 10]))
            
            ϕ_cost = Always(And(φ_ya, Implies(Negation(φ_va), Eventually(φ_va, [1, threshold]))))
            ϕ_reward = None

        if not (cfg["use_cost_suffix"] or cfg["use_cost_prefix"]):
            ϕ_cost = None
        specification = (ϕ_reward, ϕ_cost)
        print(f"specification = {specification}")

    for target_reward, target_cost in zip(args.returns, args.costs):
        seed_all(cfg["seed"])
        ret, cost, length, rew_rob, cost_rob, safe_rate = trainer.evaluate(args.eval_episodes, 
                                             target_reward*cfg["reward_scale"],
                                             target_cost*cfg["cost_scale"],
                                             specification,
                                             cfg["use_rew_prefix"],
                                             cfg["use_cost_prefix"],
                                             cfg["task"],
                                             threshold)
        env.render()
        print(
            f"Eval safe rate = {safe_rate}, Target reward {target_reward}, reward {ret}; Target cost = {target_cost}, cost {cost}"
        )

if __name__ == "__main__":
    eval()
