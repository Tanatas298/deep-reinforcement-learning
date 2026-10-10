import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNet(nn.Module):
    """Actor network (policy)
    """
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, action_size)
        # state-independent log_std
        self.log_std = nn.Parameter(torch.zeros(action_size) + 0.5)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu_logits = self.mu_head(x)
        sigma_logits = self.log_std.clamp(-20, 2).unsqueeze(0)
        return mu_logits, sigma_logits


class CriticNet(nn.Module):
    """Critic network computing the state value
    """
    def __init__(self, state_size, action_size, hidden_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value_head(x)
        return value


class ActorCriticNet(nn.Module):
    """Combining both networks and add helper methods to act and evaluate samples.
    """
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet(state_size, action_size, hidden_size)
        self.critic = CriticNet(state_size, action_size, hidden_size)
        self.eps = 1e-5

    def forward(self, x):
        x = x.reshape(1, -1)
        return self.act(x)

    def act(self, state):
        mu_logits, sigma_logits = self.actor(state)
        mu = torch.tanh(mu_logits)
        sigma = F.softplus(sigma_logits) + self.eps
        dist = Normal(mu, sigma)

        action = dist.sample()
        log_probs = dist.log_prob(action).sum(dim=-1)
        return action, log_probs

    def evaluate(self, state, action):
        mu_logits, sigma_logits = self.actor(state)
        mu = torch.tanh(mu_logits)
        sigma = F.softplus(sigma_logits) + self.eps
        dist = Normal(mu, sigma)

        log_probs = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        state_value = self.critic(state)
        return log_probs, entropy, state_value
