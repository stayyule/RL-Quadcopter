"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))        
            
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]
        self.observation_space_range = self.observation_space.high - self.observation_space.low

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]
        self.action_space_range = self.action_space.high - self.action_space.low

        # Task-specific parameters
        self.max_duration = 17.0  # secs
        # set hover target at 10.0
        self.target = np.array([0.0,0.0,10.0])
        # set initial target for target will be changed throughout time
        self.initial_target_z = self.target[2]

        self.last_pos = np.array([0.0,0.0,0.0])
        self.start = np.array([0.0,0.0,0.0])
        self.last_time = 0.0
        self.count = 0

        # flag of hover action
        self.hovered = False

        # landing set up
        self.landing_time = 5.0
        self.landing_start_time = 9.0

    def reset(self):
        self.last_pos = self.start
        self.last_time = 0.0
        self.count = 0
        self.target = np.array([0.0,0.0,10.0])
        # set initial target for target will be changed throughout time
        self.initial_target_z = self.target[2]
        # flag of hover action
        self.hovered = False
        return Pose(
                position=Point(0.0, 0.0, self.start[2]),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )


    def pos_rescale(self,state):
        #rescaling to (-5.,5.)
        mid = (self.observation_space.high + self.observation_space.low)*0.5
        rescaled = 5.0 * ((state - mid[:3])/(self.observation_space_range[:3]*0.5))
        return rescaled


    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        #position before scaling
        position = np.array([pose.position.x, pose.position.y, pose.position.z])

        scaled_pos = self.pos_rescale(position)
        scaled_tar = self.pos_rescale(self.target)
        # distance vector
        distance_vec = scaled_tar - scaled_pos
        # velocity
        if timestamp != self.last_time:
            velocity = (position - self.last_pos) / (timestamp - self.last_time)
        else:
            velocity = np.array([0.0,0.0,0.0])

        state = np.concatenate((scaled_pos, velocity, distance_vec * 5.0),axis=-1)

        self.last_pos = position
        self.last_time = timestamp

        # Compute reward / penalty and check if this episode is complete
        done = False
        
        reward_alpha = 0.3
        reward_beta = 0.05

        # distance value
        distance = np.linalg.norm(self.target - position)
        accelerate = np.linalg.norm(np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z]))

        distance_reward = (5 - distance) * reward_alpha
        accelerate_reward = accelerate * reward_beta
        reward = distance_reward - accelerate_reward

        if distance < 0.1 and self.target[2]==10.0:
            self.hovered = True
        if timestamp>self.landing_start_time and self.hovered:
            target_z = max((self.initial_target_z/self.landing_time)*(
                                    (self.landing_start_time+self.landing_time)-timestamp),0.0)
            self.target = np.array([0.0,0.0,target_z])

        print('reward={:.3} distance={:.3} target={:.3}'.format(reward ,distance_vec[2],self.target[2]),end='\r')
 

        if timestamp > self.max_duration:  # agent has run out of time
            #reward -= 10.0  # extra penalty
            done = True
        
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector
        #print("next action:", action)
        self.agent.write_sa([scaled_pos[0], scaled_pos[1], scaled_pos[2],
                velocity[2], distance_vec[2], linear_acceleration.z,
                action[2]/ 25.0,  reward])       

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:

            return Wrench(), done
