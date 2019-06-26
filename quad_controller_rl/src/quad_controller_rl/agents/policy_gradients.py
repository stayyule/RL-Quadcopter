import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.tasks.takeoff import Takeoff
from quad_controller_rl.tasks.hover import Hover

from keras import layers, models, optimizers
from keras import backend as K

import os
import pandas as pd
from quad_controller_rl import util

import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


class DDPG(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        # Task State Action
        self.task = task  # should contain observation_space and action_space

        if isinstance(self.task, Takeoff):
            self.state_size = 7
        if isinstance(self.task, Hover):
            self.state_size = 9

        self.action_size = 1

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 1000
        self.play_start_size = 100
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # for soft update of target parameters

        self.reset_episode_vars()

        self.epsilon = 1.0
        self.episode_num = 1
        self.count = 0

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

        # Save Q stats
        self.q_stats_filename = os.path.join(
            util.get_param('out'),
            "q_stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.q_stats_columns = ['episode', 'Q_value']  # specify columns to save
        print("Saving q stats {} to {}".format(self.q_stats_columns, self.q_stats_filename))  # [debug]


    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.total_q = 0.0
        self.count = 0

    def step(self, state, reward, done):

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1

        #...
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.play_start_size:
            experiences = self.memory.sample(self.batch_size)
            #print('experience:', experiences)
            self.learn(experiences)
        #...
        if done:
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward], self.stats_filename)
            self.write_stats([self.episode_num, self.total_q], self.q_stats_filename)
            self.episode_num += 1
            self.reset_episode_vars()
            #print('model:', np.array(self.actor_target.model.get_weights()[-1]).reshape(1,-1))
            print('episode ', self.episode_num, ' step count: ', self.count)

        self.last_state = state
        self.last_action = action

        # Return complete action vector
        complete_action = np.zeros(6)
        complete_action[2] = np.array(action).reshape(1)
        #print('step action:', complete_action.reshape(1,-1))
        return complete_action

    def act(self, states):
        """Returns actions [-1,1] for given state(s) as per current policy."""
        #print('states before act:', states)
        states = np.reshape(states, [-1, self.state_size])
        #print('states with shape:', states)
        actions = self.actor_local.model.predict(states)
        noise_val = self.noise.sample()
        noise_epsilon = self.epsilon
        #noise_epsilon = self.epsilon / ( int(self.episode_num / 10 ) + 1)
        #if len(self.memory) > self.batch_size:
        #    return np.around(actions + noise_epsilon * noise_val, decimals=2) # add some noise for exploration
        #else:
        #    return np.around(noise_epsilon * noise_val, decimals=2)
        final_action = np.clip(np.around(actions + noise_val * noise_epsilon, decimals=2), -1, 1)
        #print("predict:", np.around(actions, decimals=2), "noise:", np.around(noise_val, decimals=2), "action:", final_action)

        return final_action

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        self.total_q += Q_targets_next

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        #print(states, actions, Q_targets)
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) # custom training function

        # Soft-update target models
        #self.soft_update(self.critic_local.model, self.critic_target.model)
        local_weights = np.array(self.critic_local.model.get_weights())
        target_weights = np.array(self.critic_target.model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        self.critic_target.model.set_weights(new_weights)

        #self.soft_update(self.actor_local.model, self.actor_target.model)
        local_weights = np.array(self.actor_local.model.get_weights())
        target_weights = np.array(self.actor_target.model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        self.actor_target.model.set_weights(new_weights)

    def write_stats(self, stats, file_name):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(file_name, mode='a', index=False,
            header=not os.path.isfile(file_name))  # write header first time only


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        action_low (array): Min value of each action dimension
        action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        actions = layers.Dense(units=self.action_size, activation='tanh',
        name='raw_actions')(net)

        # Scale [-1, 1] output for each action dimension to proper range
        #actions = layers.Lambda(lambda x: x * self.action_range / 2,
        #name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        
        #loss = K.mean(-action_gradients * actions)

        loss = -1 * K.mean(action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss, constraints=[])
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.002)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        #print(e)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size=64):

        """Randomly sample a batch of experiences from memory."""
        #return random.sample(self.memory, k=batch_size)
        self.memory = sorted(self.memory, key=self.get_reward, reverse=True)
        return self.memory[:batch_size]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def get_reward(self, exp):
        return exp[2]