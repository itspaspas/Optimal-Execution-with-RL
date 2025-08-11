import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import copy
from collections import namedtuple, deque

from td3_model import Actor, Critic

FC1_UNITS = 24
FCS1_UNITS = 24
FC2_UNITS = 48

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_DECAY = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class TD3():
    def __init__(self, state_size, action_size, random_seed):
                self.state_size = state_size
                self.action_size = action_size
                self.seed = random.seed(random_seed)

                self.noise = OUNoise(action_size, random_seed)
                self.noise_decay = NOISE_DECAY
                
                self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
                


                # Actor Networks (local online net + target net)
                self.actor_local = Actor(state_size, action_size, random_seed).to(device)
                self.actor_target = Actor(state_size, action_size, random_seed).to(device)
                self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)

                # Critic Networks (local online net + target net)
                self.critic_local = Critic(state_size, action_size, random_seed).to(device)
                self.critic_target = Critic(state_size, action_size, random_seed).to(device)
                self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                
                self.soft_update(self.actor_local, self.actor_target, 1)
                self.soft_update(self.critic_local, self.critic_target, 1)
                
                self.learn_counter = 0
                
                
    def act(self, state, add_noise=True):

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.noise_decay
            self.noise_decay *= self.noise_decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, noise_clip=0.5, policy_freq=2):

        self.learn_counter += 1

        states, actions, rewards, next_states, dones  = experiences

        noise = torch.FloatTensor([self.noise.sample() for _ in range(len(actions))]).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)  
        next_action = (self.actor_target(next_states) + noise).clamp(-1, 1)

        target_Q1, target_Q2 = self.critic_target(next_states, next_action)
        target_Q = torch.minimum(target_Q1, target_Q2)
        target_Q = rewards + (gamma * target_Q * (1-dones)).detach()

        current_Q1, current_Q2 = self.critic_local(states, actions)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        if self.learn_counter % policy_freq == 0:
                    
                actions_pred = self.actor_local.forward(states)
                actor_loss = -self.critic_local.Q1(states, actions_pred).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.soft_update(self.actor_local, self.actor_target, TAU)
                self.soft_update(self.critic_local, self.critic_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)