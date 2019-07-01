from collections import defaultdict
import sys, os
import numpy as np
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

class TD(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space

        self.action_space = 3

        self.Q = defaultdict(lambda: np.zeros(self.action_space))

        # Episode variables
        self.reset_episode_vars()
        self.episode_num = 1
        self.step_count = 20

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        print("Saving stats to {}".format(self.stats_filename))  # [debug]

        # Save Q stats
        self.q_stats_filename = os.path.join(
            util.get_param('out'),
            "q_stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        print("Saving q stats to {}".format(self.q_stats_filename))  # [debug]

        # Save S-A stats
        self.sa_stats_filename = os.path.join(
            util.get_param('out'),
            "state_action_{}.csv".format(util.get_timestamp()))  # path to CSV file
        print("Saving states actions to {}".format(self.sa_stats_filename))  # [debug]

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.total_q = 0.0
        self.count = 0

    def step(self, state, reward, done):
            
        # Transform state vector
        state_array = state.flatten()  # convert to row vector
        #print('state:', state)
        state = ''
        for s in state_array:
            state += str(s)

        if self.count < self.step_count and self.last_action is not None:
            action = self.last_action
            self.count += 1
        else:
            action = self.act(state)
            self.count = 0
            # for h in self.Q:
            #     print('state:', h, 'arg max:', np.argmax(self.Q[h]))

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.Q[self.last_state][self.last_action] = self.update_Q(self.Q[self.last_state][self.last_action], self.Q[state][action], self.last_reward)
            self.total_reward += reward
            
        # Learn, if at end of episode
        if done:
            self.write_stats([self.episode_num, self.total_reward], ['episode', 'total_reward'], self.stats_filename)
            self.write_stats([self.episode_num, np.mean(self.total_q)], ['episode', 'Q_value'], self.q_stats_filename)
            print('total reward={:7.4f}, count={}'.format(self.total_reward, self.count))
            self.episode_num += 1
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        self.last_reward = reward

        # Return complete action vector
        complete_action = (action + 3) * 5.0
        #print('action:', action, '-', complete_action)
        return np.array([[0, 0, complete_action, 0, 0, 0]])


    def act(self, state):
        # Choose action based on given state and policy
        epsilon = int(self.episode_num / 20.0) + 1
        policy_s = self.epsilon_greedy_probs(self.Q[state], epsilon)
        action = np.random.choice(np.arange(self.action_space), p=policy_s)
        return action

    def update_Q(self, Qsa, Qsa_next, reward, alpha = 0.0001, gamma = 1):
        """ updates the action-value function estimate using the most recent time step """
        q = Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))
        self.total_q += q
        return q

    def epsilon_greedy_probs(self, Q_s, i_episode):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        policy_s = np.ones(self.action_space) * epsilon / self.action_space
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.action_space)
        #print('qs:', Q_s)
        #print('arg max:', np.argmax(Q_s))
        return policy_s

    def write_stats(self, stats, stats_columns, file_name):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=stats_columns)  # single-row dataframe
        df_stats.to_csv(file_name, mode='a', index=False,
            header=not os.path.isfile(file_name))  # write header first time only
    
    def write_sa(self,stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=['x', 'y', 'z', 'vel_z', 'tar_z', 'acce_z', 'action', 'reward'])  # single-row dataframe
        df_stats.to_csv(self.sa_stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.sa_stats_filename))  # write header first time only        
