from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class RvSTrainConfig:
    # wandb params
    project: str = "STLRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "RvS" # "R", "RC", "RCR"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # relabel_cost: bool = False
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    cost_limit: int = 0
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 200_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[1024, 1024], is_mutable=True)
    gamma: float = 1.0
    # evaluation params
    # target_returns: Tuple[Tuple[float, ...],
    #                       ...] = ((400.0, 0), (420.0, 0), (440.0, 0))  # reward, cost
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((400.0, 0.05), (420.0, 0.05), (440.0, 0.05))  # reward, cost
    eval_episodes: int = 10
    eval_every: int = 10000


@dataclass
class RvSCarCircleConfig(RvSTrainConfig):
    pass


@dataclass
class RvSAntRunConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((680.0, 0.05), (700.0, 0.05), (720.0, 0.05))


@dataclass
class RvSDroneRunConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((380.0, 0.02), (400.0, 0.02), (420.0, 0.02))


@dataclass
class RvSDroneCircleConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((620.0, 0.05), (640.0, 0.05), (660.0, 0.05))


@dataclass
class RvSCarRunConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class RvSAntCircleConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((200.0, 0.05), (220.0, 0.05), (240.0, 0.05))


@dataclass
class RvSBallRunConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((420.0, 0.02), (440.0, 0.02), (460.0, 0.02))


@dataclass
class RvSBallCircleConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((600.0, 0.05), (620.0, 0.05), (640.0, 0.05))


@dataclass
class RvSCarButton1Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSCarButton2Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSCarCircle1Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RvSCarCircle2Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RvSCarGoal1Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((36.0, 0.05), (40.0, 0.05))
    update_steps: int = 300_000
    eval_every: int = 30000


@dataclass
class RvSCarGoal2Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((36.0, 0.05), (40.0, 0.05))
    update_steps: int = 300_000
    eval_every: int = 30000


@dataclass
class RvSCarPush1Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSCarPush2Config(RvSTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSPointButton1Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSPointButton2Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSPointCircle1Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RvSPointCircle2Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RvSPointGoal1Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000
    target_returns: Tuple[Tuple[float, ...], 
                          ...] = ((20.0, 0.05), (25.0, 0.05))
    update_steps: int = 300_000
    eval_every: int = 30000


@dataclass
class RvSPointGoal2Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000
    target_returns: Tuple[Tuple[float, ...], 
                          ...] = ((20.0, 0.05), (25.0, 0.05))
    update_steps: int = 300_000
    eval_every: int = 30000


@dataclass
class RvSPointPush1Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSPointPush2Config(RvSTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RvSAntVelocityConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RvSHalfCheetahVelocityConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RvSHopperVelocityConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RvSSwimmerVelocityConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RvSWalker2dVelocityConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RvSEasySparseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSEasyMeanConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSEasyDenseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSMediumSparseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSMediumMeanConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSMediumDenseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSHardSparseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSHardMeanConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RvSHardDenseConfig(RvSTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


RvS_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": RvSCarCircleConfig,
    "OfflineAntRun-v0": RvSAntRunConfig,
    "OfflineDroneRun-v0": RvSDroneRunConfig,
    "OfflineDroneCircle-v0": RvSDroneCircleConfig,
    "OfflineCarRun-v0": RvSCarRunConfig,
    "OfflineAntCircle-v0": RvSAntCircleConfig,
    "OfflineBallCircle-v0": RvSBallCircleConfig,
    "OfflineBallRun-v0": RvSBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": RvSCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": RvSCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": RvSCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": RvSCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": RvSCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": RvSCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": RvSCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": RvSCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": RvSPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": RvSPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": RvSPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": RvSPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": RvSPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": RvSPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": RvSPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": RvSPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": RvSAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": RvSHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": RvSHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": RvSSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": RvSWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": RvSEasySparseConfig,
    "OfflineMetadrive-easymean-v0": RvSEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": RvSEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": RvSMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": RvSMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": RvSMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": RvSHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": RvSHardMeanConfig,
    "OfflineMetadrive-harddense-v0": RvSHardDenseConfig
}