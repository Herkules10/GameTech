import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha=0.0003):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Mean and standard deviation for continuous actions
        self.mean = nn.Linear(128, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        
    def forward(self, state):
        x = self.actor(state)
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha=0.0003):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        return self.critic(state)

class PPOAgent:
    def __init__(self, input_dims, n_actions, batch_size=64, alpha=0.0003,
                 n_epochs=10, horizon=2048, device='cpu', policy_clip=0.2,
                 gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.device = device
        self.horizon = horizon
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = ActorNetwork(input_dims, n_actions, alpha).to(device)
        self.critic = CriticNetwork(input_dims, alpha).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def store_transition(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def is_ready_to_learn(self):
        return len(self.states) >= self.horizon
    
    def choose_action(self, observation):
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            mean, std = self.actor(observation)
            distribution = Normal(mean, std)
            action = distribution.sample()
            prob = distribution.log_prob(action).sum()
            value = self.critic(observation)
            
            # Ensure action has correct shape for environment
            action = action.reshape(-1)
            
        self.actor.train()
        self.critic.train()
        return action, prob, value
        
    def calculate_advantages(self):
        advantages = []
        returns = []
        gae = 0
        
        # Convert lists to tensors
        values = torch.tensor(self.vals, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        
        # Calculate GAE and returns
        for t in reversed(range(len(self.rewards)-1)):
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    def learn(self):
        for _ in range(self.n_epochs):
            # Calculate advantages and returns
            advantages, returns = self.calculate_advantages()
            
            # Convert memory to tensors
            states = torch.stack(self.states[:-1])  # Exclude last state
            old_probs = torch.stack(self.probs[:-1])
            actions = torch.stack(self.actions[:-1])
            
            # Create mini-batches
            batch_start = np.arange(0, len(states), self.batch_size)
            indices = np.arange(len(states), dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i:i+self.batch_size] for i in batch_start]
            
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current action probabilities and values
                mean, std = self.actor(batch_states)
                distribution = Normal(mean, std)
                new_probs = distribution.log_prob(batch_actions).sum(dim=-1)
                critic_value = self.critic(batch_states).squeeze()
                
                # Calculate ratios and surrogate losses
                prob_ratio = (new_probs - batch_old_probs).exp()
                
                # Calculate actor loss using clipped surrogate objective
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Calculate critic loss
                critic_loss = nn.MSELoss()(critic_value, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []