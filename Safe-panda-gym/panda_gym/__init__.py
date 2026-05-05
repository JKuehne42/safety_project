import os
from panda_gym.envs.panda_tasks.panda_tasks import (
    PandaFlipEnv,
    PandaPickAndPlaceEnv,
    PandaPushEnv,
    PandaReachEnv,
    PandaSlideEnv,
    PandaStackEnv,
)
from panda_gym.envs.panda_tasks.panda_pick_and_place_platform import PandaPickAndPlacePlatformEnv
from panda_gym.envs.panda_tasks.panda_stack_pyramid import PandaStackPyramidEnv
from panda_gym.envs.panda_tasks.panda_stack_3 import PandaStack3Env
from panda_gym.envs.panda_tasks.panda_build_L import PandaBuildLEnv
from panda_gym.envs.panda_tasks.panda_push_safe import PandaPushSafeEnv
from panda_gym.envs.panda_tasks.panda_reach_safe import PandaReachSafeEnv
from panda_gym.envs.panda_tasks.panda_slide_safe import PandaSlideSafeEnv
from panda_gym.envs.panda_tasks.panda_pick_and_place_safe import PandaPickAndPlaceSafeEnv
from panda_gym.envs.panda_tasks.panda_stack_safe import PandaStackSafeEnv

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip",
             "ReachSafe", "PushSafe", "SlideSafe", "PickAndPlaceSafe", "StackSafe",
             "Stack3", "StackPyramid", "PickAndPlacePlatform", "BuildL"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Panda{task}{control_suffix}{reward_suffix}-v2"

            # FIXED: Updated max_episode_steps for all stacking tasks
            if task in ["Stack", "Stack3", "StackPyramid", "StackSafe", "BuildL"]:
                max_steps = 100
            else:
                max_steps = 50

            register(
                id=env_id,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=max_steps,
            )

            ENV_IDS.append(env_id)