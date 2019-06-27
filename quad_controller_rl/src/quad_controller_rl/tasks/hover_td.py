"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover_TD(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))        
            
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.scale = 15.0

    def reset(self):

        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)

        scaled_x = pose.position.x / self.scale * 5.0
        scaled_y = pose.position.y / self.scale * 5.0
        scaled_z = pose.position.z / self.scale * 5.0

        del_x = (self.target_x - pose.position.x) / self.scale * 5.0
        del_y = (self.target_y - pose.position.y) / self.scale * 5.0
        del_z = (self.target_z - pose.position.z) / self.scale * 5.0

        state = np.around(np.array([scaled_x, scaled_y, scaled_z]), decimals=0)

        # Compute reward / penalty and check if this episode is complete
        done = False

        distance = np.power(np.power(del_x, 2) + np.power(del_y, 2) + np.power(del_z, 2), 0.5)

        reward = 5.0 - distance
        

        if timestamp > self.max_duration:  # agent has run out of time
            done = True

        if pose.position.z >= 15:
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector
        #print("next action:", action)
        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:

            return Wrench(), done



