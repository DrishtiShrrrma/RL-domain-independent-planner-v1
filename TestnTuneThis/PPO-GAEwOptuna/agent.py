
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, MultivariateNormal
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, seed=None):
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        if isinstance(action_size, gym.spaces.Discrete):
            self.actor = nn.Linear(64, action_size.n)
        elif isinstance(action_size, gym.spaces.Box):
            self.actor_mean = nn.Linear(64, action_size.shape[0])
            self.actor_std = nn.Linear(64, action_size.shape[0])

        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if isinstance(self.action_size, gym.spaces.Discrete):
            action_probs = F.softmax(self.actor(x), dim=-1)
        elif isinstance(self.action_size, gym.spaces.Box):
            mean = self.actor_mean(x)
            std = F.softplus(self.actor_std(x))
            action_probs = mean, std

        value = self.critic(x)

        return action_probs, value

class PPOAgent:
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_size = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_size = action_space

        self.model = ActorCritic(action_space.shape[0], self.action_size, seed=seed).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

        self.memory = []
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.lmbda = 0.95
        self.eps = np.finfo(np.float32).eps.item()

    def initialize(self, state_size):
        self.model = ActorCritic(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def sample_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std, _ = self.model(state)

            if isinstance(self.action_space, gym.spaces.Discrete):
                dist = Categorical(mean)
                action = dist.sample().item()
            elif isinstance(self.action_space, gym.spaces.Box):
                dist = MultivariateNormal(mean, torch.diag_embed(std))
                action = dist.sample().cpu().numpy()[0]

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        states = torch.tensor([t[0] for t in self.memory], dtype=torch.float32).to(self.device)
        actions = torch.tensor([t[1] for t in self.memory], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([t[2] for t in self.memory], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([t[3] for t in self.memory], dtype=torch.float32).to(self.device)
        dones = torch.tensor([t[4] for t in self.memory], dtype=torch.float32).to(self.device)

        _, next_value = self.model(next_states[-1])
        next_value = next_value.detach()

        returns = []
        advantages = []
        advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lmbda * (1 - dones[t]) * advantage
            next_value = values[t]
            returns.insert(0, next_value + advantage)
            advantages.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(10):  # Update the model for 10 epochs
            action_probs, values = self.model(states)

            if isinstance(self.action_space, gym.spaces.Discrete):
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(actions)
            elif isinstance(self.action_space, gym.spaces.Box):
                dist = MultivariateNormal(action_probs[0], torch.diag_embed(action_probs[1]))
                log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(log_probs - log_probs.detach())

            actor_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean()
            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []
