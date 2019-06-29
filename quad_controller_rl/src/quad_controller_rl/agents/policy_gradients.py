import numpy as np
import os
import pandas as pd

from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

from keras import layers, models, optimizers
from keras import backend as K


# Create DDPG Actor
class Actor:

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize Parameters and build model"""
        self.state_size = state_size  # integer - dimension of each state
        self.action_size = action_size  # integer - dimension of each action
        self.action_low = action_low  # array - min value of action dimension
        self.action_high = action_high  # array - max value of action dimension
        self.action_range = self.action_high - self.action_low

        self.build_model()

    def build_model(self):
        """Create actor network that maps states to actions"""

        # Define input states
        states = layers.Input(shape=(self.state_size,), name='states')

        # Create hidden layers
        net = layers.Dense(units=64, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        #net = layers.Dense(units=32, activation='relu')(net)

        # Output layer with sigmoid
        actions =layers.Dense(units=self.action_size, activation='tanh', name='actions',kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))(net)

        # Scale output for each action to appropriate ranges
        #actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create model
        self.model = models.Model(inputs=states, outputs=actions)

        # Loss function using Q-val gradients
        action_grads = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_grads * actions)

        # Optimizer and Training Function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_grads, K.learning_phase()], outputs=[], updates=updates_op)


# Create DDPG Critic
class Critic:

    def __init__(self, state_size, action_size):
        """Initialize parameters and model"""

        self.state_size = state_size  # integer - dim of states
        self.action_size = action_size  # integer - dim of action

        self.build_model()

    def build_model(self):
        """Critic network for mapping state-action pairs to Q-vals"""

        # Define inputs
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layers for states
        net_states = layers.Dense(units=64, activation='relu')(states)
        #net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Hidden layers for actions
        #net_actions = layers.Dense(units=64, activation='relu')(actions)

        # Concatenate state and action values
        net = layers.Concatenate(axis=-1)([net_states, actions])
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=64, activation='relu')(net)
        # Output layer for Q-values
        Q_vals = layers.Dense(units=1,name='q_vals',kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))(net)

        # Create model
        self.model = models.Model(inputs=[states, actions], outputs=Q_vals)

        # Define Optimizer and compile
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute Q' wrt actions
        action_grads = K.gradients(Q_vals, actions)

        # Create function to get action grads
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_grads)


class DDPG(BaseAgent):

    def __init__(self, task):

        self.task = task

        # Constrain State and Action matrices
        self.state_size = 9
        self.action_size = 1
        # For debugging:
        print("Constrained State {} and Action {}; Original State {} and Action {}".format(self.state_size, self.action_size,
            self.task.observation_space.shape, self.task.action_space.shape))


        # Save episode statistics for analysis
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward']
        self.episode_num = 1
        print("Save Stats {} to {}".format(self.stats_columns, self.stats_filename))

        # Actor Model
        self.acts=np.zeros(shape=self.task.action_space.shape)
        self.action_low = self.task.action_space.low[2]
        self.action_high = self.task.action_space.high[2]
        self.action_range=self.task.action_space.high[2]-self.task.action_space.low[2]
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize model parameters with local parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Process noise
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size)

        # ALGORITHM PARAMETERS
        self.gamma = 0.99  # discount
        self.tau = 0.005  # soft update of targets
        self.count=0

        # Episode vars
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.acts = np.zeros(shape=self.task.action_space.shape)

    def step(self, state, reward, done):

        # Choose an action
        action = self.act(state)
        self.count += 1
        # Save experience and reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward

        if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
        if done:
            # Learn from memory

            # Write episode stats and reset
            self.write_stats([self.episode_num, self.total_reward])
            print('total reward={:7.4f}, count={}'.format(self.total_reward, self.count))
            self.episode_num += 1
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        self.acts[2]=action*25.0
        # Returns completed action vector
        return self.acts

    def act(self, states):
        """Returns actions for a given state for current policy"""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()

    def learn(self, experiences):
        """Update policy and value parameters given experiences"""
        # Convert experiences to separate arrays for each element
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next actions and Q-vals from target model
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current state and train critic
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target model
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model params"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def write_stats(self, stats):
        """Write an episode of stats to a CSV file"""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))

    def preprocess(self, state, state_size=3):
        """Return state vector of just linear position and velocity"""
#         state = np.concatenate(state[0:3], state[7:10])
        state = state[0:3]
        return state

    def postprocess(self, action):
        """Return action vector of linear forces by default"""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[2] = action
        return complete_action

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
        return random.sample(self.memory, k=batch_size)
        # self.memory = sorted(self.memory, key=self.get_reward, reverse=True)
        # return self.memory[:batch_size]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def get_reward(self, exp):
        return exp[2]