import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent

from keras import layers, models, optimizers
from keras import backend as K

import os
import pandas as pd
from quad_controller_rl import util

import random

from collections import namedtuple

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])


class DDPG_Takeoff(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        # Task State Action
        self.task = task  # should contain observation_space and action_space

        self.state_size = 7
        self.action_size = 1

        # Actor (Policy) Model
        self.acts=np.zeros(shape=self.task.action_space.shape)
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
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # for soft update of target parameters
        self.count=0

        self.reset_episode_vars()

        self.epsilon = 1
        self.episode_num = 1

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
        self.total_reward = 0.0
        self.total_q = 0.0
        self.count = 0
        self.acts = np.zeros(shape=self.task.action_space.shape)


    def step(self, state, reward, done):

        # Choose an action
        action = self.act(state)
        self.count += 1
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward

        #...
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            #print('experience:', experiences)
            self.learn(experiences)
        else:
            action = np.array([0.8]).reshape(1,-1)

        if done:
            # Write episode stats
            # 241
            #print('episode ', self.episode_num, ' step count: ', self.count)
            self.write_stats([self.episode_num, self.total_reward], ['episode', 'total_reward'], self.stats_filename)
            self.write_stats([self.episode_num, np.mean(self.total_q)], ['episode', 'Q_value'], self.q_stats_filename)
            print('total reward={:7.4f}, count={}'.format(self.total_reward, self.count))
            self.episode_num += 1
            self.reset_episode_vars()
            #print('model:', np.array(self.actor_target.model.get_weights()[-1]).reshape(1,-1))

        self.last_state = state
        self.last_action = action

        self.acts[2]=action*25.0
        # Returns completed action vector
        return self.acts

    def act(self, states):
        """Returns actions [-1,1] for given state(s) as per current policy."""
        #print('states before act:', states)
        states = np.reshape(states, [-1, self.state_size])
        #print('states with shape:', states)
        actions = self.actor_local.model.predict(states)
        noise_val = self.noise.sample()
        #noise_epsilon = self.epsilon
        noise_epsilon = self.epsilon / ( int(self.episode_num / 30 ) + 1)
        #if len(self.memory) > self.batch_size:
        #    return np.around(actions + noise_epsilon * noise_val, decimals=2) # add some noise for exploration
        #else:
        #    return np.around(noise_epsilon * noise_val, decimals=2)
        #tanh
        final_action = np.clip(np.around(actions + noise_val * noise_epsilon, decimals=2), -1, 1)
        #sigmoid
        #final_action = np.clip(np.around(actions + noise_val * noise_epsilon, decimals=2), 0, 1)
        #print("predict:", np.around(actions, decimals=2), "noise:", np.around(noise_val * noise_epsilon, decimals=2), "action:", np.around(final_action,decimals=2))

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
        #print('Q next:', Q_targets_next)
        self.total_q += Q_targets_next

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        #print(states, actions, Q_targets)
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        #print('action_gradients:', action_gradients)
        self.actor_local.train_fn([states, action_gradients, 1]) # custom training function

        # Soft-update target model
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

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

    def soft_update(self, local_model, target_model):
        """Soft update model params"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

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
        self.hidden_layer1 = 64
        self.hidden_layer2 = 64
        self.learning_rate = 0.0001

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=self.hidden_layer1, activation='relu')(states)
        net = layers.Dense(units=self.hidden_layer2, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        # -----kernel
        actions = layers.Dense(units=self.action_size, activation='tanh',
        name='raw_actions',
        kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))(net)

        # Scale [-1, 1] output for each action dimension to proper range
        #actions = layers.Lambda(lambda x: x * self.action_range / 2,
        #name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))

        loss = K.mean( action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
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
        self.hidden_layer1 = 64
        self.hidden_layer2 = 64
        self.learning_rate = 0.001
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=self.hidden_layer1, activation='relu')(states)

        # Concatenate state and action values
        net = layers.Concatenate(axis=-1)([net_states, actions])
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=self.hidden_layer2, activation='relu')(net)

        # Add more layers to the combined network if needed


        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values',
        kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    #0.15 0.3
    def __init__(self, size, mu=None, theta=0.15, sigma=0.02, dt=1e-2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Circular buffer for storing experience tuples"""
    
    def __init__(self, size=1000):
        """Initialize ReplayBuffer"""
        self.size = size
        self.memory = []
        self.idx = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory"""
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[idx] = e
            self.idx = (self.idx +1) % self.size
            
    def sample(self, batch_size=64):
        """Random sample of experiences"""
        return random.sample(self.memory, k=batch_size)
    
    def __len__(self):
        """Return size of internal memory"""
        return len(self.memory)

