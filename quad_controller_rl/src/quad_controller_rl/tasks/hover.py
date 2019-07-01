"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
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
        self.max_duration = 8.0  # secs
        self.takeoff_duration = 5.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff

        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0


        self.scale = 15.0

        self.final_target = 10.0
        self.last_time = 0.0
        self.last_action = None

    def reset(self):
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        self.action = None
        self.last_action = None
        self.last_time = 0.0
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
        scaled_z = pose.position.z / self.scale

        # vel_x = pose.position.x - self.last_x
        # vel_y = pose.position.y - self.last_y
        # vel_z = pose.position.z - self.last_z
        print('timestamp:', timestamp, 'last time:', self.last_time)
        if timestamp != self.last_time:
            vel_x = (pose.position.x - self.last_x) / (timestamp - self.last_time) / self.scale
            vel_y = (pose.position.y - self.last_y) / (timestamp - self.last_time) / self.scale
            vel_z = (pose.position.z - self.last_z) / (timestamp - self.last_time) / self.scale
        else:
            vel_x = vel_y = vel_z = 0.0
        
        del_x = (self.target_x - pose.position.x) / self.scale
        del_y = (self.target_y - pose.position.y) / self.scale
        del_z = (self.target_z - pose.position.z) / self.scale

        # del_x = self.target_x - pose.position.x
        # del_y = self.target_y - pose.position.y
        # del_z = self.target_z - pose.position.z

        state = np.around(np.array([
                scaled_x, scaled_y, scaled_z,
                vel_z,
                del_z,
                linear_acceleration.z,
                timestamp / 10.0 ]), decimals=2)
        # state = np.around(np.array([
        #         scaled_x, scaled_y, scaled_z,
        #         vel_x * 10.0, vel_y * 10.0, vel_z * 10.0,
        #         del_z ]), decimals=2)
        # state = np.around(np.array([
        #         pose.position.x, pose.position.y, pose.position.z,
        #         linear_acceleration.x, linear_acceleration.y, linear_acceleration.z,
        #         del_z ]), decimals=2)

        self.last_x = pose.position.x
        self.last_y = pose.position.y
        self.last_z = pose.position.z
        self.last_time = timestamp

        # Compute reward / penalty and check if this episode is complete
        done = False
        
        reward_alpha = 0.5
        reward_beta = 0.5

        #distance = np.linalg.norm([del_x, del_y, del_z])
        distance = min(abs(self.target_z - pose.position.z), 10.0)

        distance_reward = (10 - distance) * reward_alpha
        # accelerate_reward = abs(vel_z) * reward_beta
        if self.last_action is not None:
            accelerate_reward = abs(self.last_action) * reward_beta
        else:
            accelerate_reward = 0

        #reward = distance_reward - accelerate_reward
        if pose.position.z <= 5.0:
            reward = distance_reward
        else:
            reward = distance_reward - accelerate_reward

        print('==========')
        print('height:', pose.position.z)
        print('reward:', reward)
        print('accelerate:', accelerate_reward)
        
        # target_val = max(self.final_target / self.takeoff_duration * timestamp, 5)

        # self.target_z = min (np.around(target_val), 10.0)
        # print('target:', self.target_z)

        if timestamp > self.max_duration:  # agent has run out of time
            #reward -= 10.0  # extra penalty
            done = True

        if pose.position.z > 20.0:
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector
        self.last_action = action / 25.0

        self.agent.write_sa([pose.position.x, pose.position.y, pose.position.z,
                        vel_z, self.target_z - pose.position.z, linear_acceleration.z,
                        action[2], reward])


        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:

            return Wrench(), done
