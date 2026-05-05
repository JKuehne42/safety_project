from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core_safe import Task
from panda_gym.utils import distance


class Stack(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="dense",
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04

        self.get_ee_position = get_ee_position
        # unsafe parameters
        self.unsafe_region_radius = 0.1
        self.max_x_base_unsafe = 0.25 # max x pos to put unsafe space
        self.min_x_base_unsafe = -0.25
        self.max_y_base_unsafe_boundary = 0.25
        # unsafe parameters
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

        self.sim.create_sphere(
            body_name = "unsafe_region",
            radius = self.unsafe_region_radius,
            mass = 0.0,
            ghost = True,
            position = np.array([0.0, 0.0, -self.object_size / 2]),
            rgba_color = np.array([0.9, 0.1, 0.1, 0.3]),
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

        unsafe_space = self.sim.get_base_position("unsafe_region")
        end_effector_pos = self.get_end_effector_position().copy()

        target_position1 = np.array(self.sim.get_base_position("target1"))
        target_position2 = np.array(self.sim.get_base_position("target2"))
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
                unsafe_space,
                np.array([self.unsafe_region_radius]), # unsafe region radius or size
                    # add target position and possibly orientation
                target_position1,
                target_position2,
                end_effector_pos
            ]
        )
        return observation

    def get_end_effector_position(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def _sample_unsafe_state_z_boundary(self):
        # height boundary constraint
        return np.random.uniform(0.28, - self.object_size / 2 )

    def _sample_unsafe_state_x_boundary(self):
        # x boundary constraint
        return np.random.uniform(self.max_x_base_unsafe,  self.min_x_base_unsafe )

    def _sample_unsafe_state_y_boundary(self):
        return np.random.uniform(self.max_y_base_unsafe_boundary, -self.max_y_base_unsafe_boundary )

    def _sample_unsafe_state(self):
        # sample boundary to right after base of robot    
        random_x_base = self._sample_unsafe_state_x_boundary()
        random_z_base = self._sample_unsafe_state_z_boundary()
        randome_y_base = self._sample_unsafe_state_y_boundary()
        return np.array([random_x_base, randome_y_base ,random_z_base ])

    def _target_in_unsafe_region(self, target_pos):
        # target_pos = self.goal
        # distance between object anf unsafe space
        distance_target_unsafe = distance(target_pos , self.unsafe_state_pos )
       
        # lowest distance possible is sphere radius plus max posssible value using size 
        # this is a mathematical apporximation
        min_distance_treshhold = self.unsafe_region_radius + (self.object_size / 2)
        return distance_target_unsafe < min_distance_treshhold

    def _object_in_unsafe_region(self, object_pos):
        # object_pos = self.get_achieved_goal()
        # distance between object anf unsafe space
        distance_obj_unsafe = distance(object_pos , self.unsafe_state_pos )
        
        # lowest distance possible is sphere radius plus max posssible value using size 
        # this is a mathematical apporximation
        min_distance_treshhold = self.unsafe_region_radius + (self.object_size / 2)
        return (distance_obj_unsafe < min_distance_treshhold)

    def _end_effector_in_unsafe_region(self):
        end_effector_pos = self.get_end_effector_position()
        # distacne between end effector and unsafe space center
        distance_end_effector_unsafe = distance(end_effector_pos , self.unsafe_state_pos )
       
        min_distance_treshhold = self.unsafe_region_radius
        return (distance_end_effector_unsafe < min_distance_treshhold)


    def reset(self) -> None:
        self.unsafe_state_pos = self._sample_unsafe_state()
        self.sim.set_base_pose("unsafe_region",
                            self.unsafe_state_pos,
                            np.array([0.0, 0.0, 0.0, 1.0]))

        self.goal = self._sample_goal()
        
        if self._target_in_unsafe_region(self.goal[:3]) or \
            self._target_in_unsafe_region(self.goal[3:]):
            self.goal = self._sample_goal()
        
        object1_position, object2_position = self._sample_objects()

        if self._object_in_unsafe_region(object1_position) or \
            self._object_in_unsafe_region(object2_position):
            object1_position, object2_position = self._sample_objects()



        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))



        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))


    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        goal2 += noise
        return np.concatenate((goal1, goal2))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, 3 * self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position

    def _object_not_on_table(self):
        object_pos = self.get_achieved_goal()
        # distance between object anf unsafe space
        z_pos1 = object_pos[ :3][2]
        z_pos2 = object_pos[3: ][2]
        return z_pos1 < 0.0 or z_pos2<0.0

    def _compute_cost_function(self):
        '''
        two possible contontitions to get a non 0 cost
        1 Object enters unsafe space 1  or 2
        2 end effector enters unsafe space 1 or 2
        '''
        object_pos = self.get_achieved_goal()
        if self._object_in_unsafe_region(object_pos[:3]) or\
            self._object_in_unsafe_region(object_pos[3:]):
            return 1.0
        elif self._end_effector_in_unsafe_region():
            return 1.0
        elif self._object_not_on_table():
            return 2.0
        else:
            return 0.0

    def compute_cost(self) -> Union[np.ndarray, float]:
        unsafe_cost = self._compute_cost_function()
        return unsafe_cost

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        d = distance(achieved_goal, desired_goal)
        return np.array((d < self.distance_threshold), dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array((d > self.distance_threshold), dtype=np.float64)
        else:
            return -d
