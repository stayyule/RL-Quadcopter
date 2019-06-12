"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, delta position_x, .._y, .._z, linear_acceration_x, .._y, .._z>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, - cube_size, - cube_size, - cube_size,           0.0,            0.0,            0.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,   cube_size,   cube_size,   cube_size, cube_size / 2,  cube_size / 2,  cube_size / 2]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        self.scale = cube_size / 2


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

        scaled_x = pose.position.x / self.scale
        scaled_y = pose.position.y / self.scale
        scaled_z = pose.position.z / self.scale - 1
        scaled_x *= 5
        scaled_y *= 5
        scaled_z *= 5
        target_x = self.target_x / self.scale
        target_y = self.target_y / self.scale
        target_z = self.target_z / self.scale - 1
        target_x *= 5
        target_x *= 5
        target_z *= 5
        del_x = target_x - scaled_x
        del_y = target_y - scaled_y
        del_z = target_z - scaled_z

        state = np.array([
                scaled_x, scaled_y, scaled_z,
                (scaled_x - self.last_x)*10, (scaled_y - self.last_y)*10, (scaled_z - self.last_z)*10,
                del_x, del_y, del_z])
        #print('state', state)

        self.last_x = scaled_x
        self.last_y = scaled_y
        self.last_z = scaled_z

        # Compute reward / penalty and check if this episode is complete
        done = False
        
        reward_alpha = 0.1
        reward_beta = 0.02

        distance = np.power(np.power(del_x,2) + np.power(del_y,2) + np.power(del_z,2), 0.5)
        accel = np.power(np.power(linear_acceleration.x,2) + np.power(linear_acceleration.y,2) + np.power(linear_acceleration.z,2), 0.5)

        distance_reward = (5.0 - distance) * reward_alpha
        accelerate_reward = accel * reward_beta

        reward = distance_reward - accelerate_reward

        print('reward:', reward)
        print('distance:', distance_reward)
        print('accelerate:', accelerate_reward)

        if pose.position.z > 20.0:
            reward -= 10.0  # extra penalty
            done = True

        if timestamp > self.max_duration:  # agent has run out of time
            reward -= 10.0  # extra penalty
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
