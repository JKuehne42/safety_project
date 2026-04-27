<h1 align="center">
<br>
Temporal Logic Specification-Conditioned Decision Transformer for Offline Safe Reinforcement Learning
</h1>

<p align="center">
Repo for "<a href="https://proceedings.mlr.press/v235/guo24j.html" target="_blank">SDT: <u>S</u>pecification-conditioned <u>D</u>ecision <u>T</u>ransformer</a>" [ICML 2024]
</p>

## Method

**SDT** is an offline RL framework that integrates signal temporal logic (STL) for specifying complex temporal rules with the sequential modeling of Decision Transformer (DT). It conditions on STL robustness values to learn safe, high-reward policies and to adapt to varying levels of specification satisfaction.

## Installation
The code is adapted from [Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym), [Decision Transformer](https://github.com/kzl/decision-transformer), and [OSRL](https://github.com/liuzuxin/OSRL). To install the packages, please refer to the respective README.md in each folder.


## Training

To train SDT and the baselines, use the following commands:
```bash
# SDT
python examples/train/train_cdt.py --task <env-name> \
    --use_cost_prefix True --use_cost_suffix True --reward_scale 0.001 --other_params ...

# CDT
python examples/train/train_cdt.py --task <env-name> --other_params ...

# RvS-RC
python examples/train/train_rvs.py --task <env-name> --prefix RvS-RC --other_params ...

# RvS-Rρ
python examples/train/train_rvs.py --task <env-name> --prefix RvS-RCR \
    --reward_scale 0.001 --other_params ...

# Other baselines
python examples/train/train-<method>.py --task <env-name> --other_params ...
```

To train DT and DT with reward prefix, run:
```bash
# DT
python experiment.py --env <env-name> --dataset <data-type> --model_type dt --wandb True

# DT with reward prefix
python experiment.py --env <env-name> --dataset <data-type> --model_type dt \
    --wandb True --use_prefix True
```


## Evaluation

To evaluate a trained model, run:
```bash
python examples/eval/eval_<method>.py --path <path-to-model> --other_params ...
```

## Bibtex

If you find our work useful, please cite it as:
```
@inproceedings{guo2024temporal,
  title={Temporal Logic Specification-Conditioned Decision Transformer for Offline Safe Reinforcement Learning},
  author={Guo, Zijian and Zhou, Weichao and Li, Wenchao},
  booktitle={International Conference on Machine Learning},
  pages={17003--17019},
  year={2024},
  organization={PMLR}
}
```