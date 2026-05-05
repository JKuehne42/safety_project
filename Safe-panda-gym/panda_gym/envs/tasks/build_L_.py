from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class  BuildL(Task):
    def __init__(
        self,
        sim : PyBullet,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="object3",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target3",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

        self.sim.create_box(
            body_name="object4",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target4",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )


    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))

        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))

        object3_position = np.array(self.sim.get_base_position("object3"))
        object3_rotation = np.array(self.sim.get_base_rotation("object3"))
        object3_velocity = np.array(self.sim.get_base_velocity("object3"))
        object3_angular_velocity = np.array(self.sim.get_base_angular_velocity("object3"))

        object4_position = np.array(self.sim.get_base_position("object4"))
        object4_rotation = np.array(self.sim.get_base_rotation("object4"))
        object4_velocity = np.array(self.sim.get_base_velocity("object4"))
        object4_angular_velocity = np.array(self.sim.get_base_angular_velocity("object4"))


        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,

                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,

                object3_position,
                object3_rotation,
                object3_velocity,
                object3_angular_velocity,

                object4_position,
                object4_rotation,
                object4_velocity,
                object4_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        object3_position = self.sim.get_base_position("object3")
        object4_position = self.sim.get_base_position("object4")

        
        achieved_goal = np.concatenate((object1_position, object2_position, object3_position, object4_position))
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position, object2_position, object3_position, object4_position = self._sample_objects()

        self.sim.set_base_pose("target1", self.goal[ :3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:6], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target3", self.goal[6:9], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target4", self.goal[9: ], np.array([0.0, 0.0, 0.0, 1.0]))
        
        
        self.sim.set_base_pose("object1",  object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2",  object2_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object3",  object3_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object4",  object4_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        z_pos =  self.object_size / 2
        goal1 = np.array([0.0, 0.0, z_pos])  # z offset for the cube center
        goal2 = np.array([0.0, 2 * self.object_size / 2, z_pos])  # y offset for the cube center
        goal3 = np.array([0.0, 4 * self.object_size / 2, z_pos])  # y offset for the cube center
        goal4 = np.array([self.object_size, 4 * self.object_size / 2, z_pos])  # y offset for the cube center

        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

        goal1 += noise
        goal2 += noise
        goal3 += noise
        goal4 += noise

        return np.concatenate((goal1, goal2, goal3, goal4))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, self.object_size / 2])
        object3_position = np.array([0.0, 0.0, self.object_size / 2])
        object4_position = np.array([0.0, 0.0, self.object_size / 2])

        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise3 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise4 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        
        object1_position += noise1
        object2_position += noise2
        object3_position += noise3
        object4_position += noise4
        
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position, object3_position, object4_position

    def _get_object_orietation(self):
        object1_rotation = np.array(self.sim.get_base_rotation("object1", "quaternion"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2", "quaternion"))
        object3_rotation = np.array(self.sim.get_base_rotation("object3", "quaternion"))
        object4_rotation = np.array(self.sim.get_base_rotation("object4", "quaternion"))
        return object1_rotation, object2_rotation, object3_rotation, object4_rotation

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=np.float64)
    def compute_cost(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = ...) -> Union[np.ndarray, float]:
        objective_position = self.goal
        current_position = self.get_achieved_goal()


        cost1 = distance(objective_position[:3], current_position[:3])
        cost2 = distance(objective_position[3:6], current_position[3:6])
        cost3 = distance(objective_position[6:9], current_position[6:9])
        cost4 = distance(objective_position[9: ], current_position[9: ])

        
        
        cost = {"cost1": cost1, "cost2": cost2, "cost3": cost3, "cost4": cost4}

        if self.reward_type == "sparse":
            cost_sparse = cost.copy()
            for key in cost_sparse:
                sparse_co = np.array((cost_sparse[key] > self.distance_threshold), dtype=np.float64)
                cost_sparse[key] = sparse_co
            return cost_sparse
        else:
            return cost 

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
