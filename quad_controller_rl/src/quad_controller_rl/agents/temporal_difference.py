from collections import defaultdict
import sys, os
import numpy as np
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

class TD(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space

        self.state_space = 15
        self.action_space = 50

        self.Q = defaultdict(lambda: np.zeros(slef.action_space))

        # Episode variables
        self.reset_episode_vars()
        self.episode_num = 1

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save

        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):
        # Transform state vector
        state = state.reshape(1, -1)[2]  # convert to row vector
        print('state:', state)

        # Choose an action
        action = self.act(state)
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.Q[last_state][last_action] = self.update_Q(self.Q[last_state][last_action], self.Q[state][action], last_reward)
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            self.Q[last_state][last_action] = self.update_Q(self.Q[last_state][last_action], 0, last_reward)
            self.reset_episode_vars()
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

        self.last_state = state
        self.last_action = action
        self.last_reward = rewardzzz

        # Return complete action vector
        complete_action = action - 25
        #print('action', action)
        complete_action[0][-3:] = np.zeros(3) # linear force only
        complete_action[0][:2] = np.zeros(2) # z only
        return complete_action


    def act(self, state):
        # Choose action based on given state and policy
        policy_s = self.epsilon_greedy_probs(self.Q[state], self.episode_num)
        action = np.random.choice(np.arrange(self.action_space), p=policy_s)
        return action

    def update_Q(Qsa, Qsa_next, reward, alpha = 0.001, gamma = 0.99):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(Q_s, i_episode):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        policy_s = np.ones(self.action_space) * epsilon / self.action_space
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.action_space)
        return policy_s

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only
