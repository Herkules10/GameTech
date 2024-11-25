import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PPOMemory:
    def __init__(self, horizon):
        self.horizon = horizon
        self.clear_memory()
        
    def generate_batches(self, batch_size, device):
        n_states = len(self.states)
        indices = np.random.permutation(n_states)
        batch_starts = range(0, n_states, batch_size)
        batches = [indices[i:i+batch_size] for i in batch_starts]
        
        # Convert lists to tensors only when generating batches
        states_tensor = torch.FloatTensor(self.states).to(device)
        actions_tensor = torch.FloatTensor(self.actions).to(device)
        probs_tensor = torch.FloatTensor(self.probs).to(device)
        vals_tensor = torch.FloatTensor(self.vals).to(device)
        rewards_tensor = torch.FloatTensor(self.rewards).to(device)
        dones_tensor = torch.BoolTensor(self.dones).to(device)
        
        return states_tensor, actions_tensor, probs_tensor, vals_tensor, \
               rewards_tensor, dones_tensor, batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        # Convert tensors to lists/numpy arrays when storing
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(vals, torch.Tensor):
            vals = vals.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def is_ready(self):
        return len(self.states) >= self.horizon

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, device,
                 fc1_dims=64, fc2_dims=64, fc3_dims=64):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, n_actions),
            nn.Tanh()
        ).to(device)
        
        self.log_std = nn.Parameter(torch.zeros(n_actions).to(device))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        
    def forward(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, device, fc1_dims=64, fc2_dims=64, fc3_dims=64):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, 1)
        ).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        
    def forward(self, state):
        return self.critic(state)

class PPOAgent:
    def __init__(self, input_dims, n_actions, device, alpha=0.0003, gamma=0.99,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, horizon=2048):
        self.device = device
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.horizon = horizon
        
        self.actor = ActorNetwork(input_dims, n_actions, alpha, device)
        self.critic = CriticNetwork(input_dims, alpha, device)
        self.memory = PPOMemory(horizon)
        
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor.optimizer, step_size=10, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=10, gamma=0.9)
        
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
        
    def choose_action(self, observation):
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            dist = self.actor(state)
            action = dist.sample()
            probs = torch.exp(dist.log_prob(action).sum(dim=-1))
            value = self.critic(state)
            
            return action.cpu().numpy()[0], probs.cpu().numpy(), value.cpu().numpy()
    
    def learn(self):
        if not self.memory.is_ready():
            return
            
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, done_arr, batches = \
                self.memory.generate_batches(self.batch_size, self.device)
                
            values = vals_arr
            advantage = self.compute_gae(reward_arr, values, done_arr)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            for batch in batches:
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]
                
                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                
                new_probs = torch.exp(dist.log_prob(actions).sum(dim=-1))
                prob_ratio = new_probs / old_probs
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        current_actor_lr = self.actor.optimizer.param_groups[0]['lr']
        current_critic_lr = self.critic.optimizer.param_groups[0]['lr']
        print(f"Learning rates - Actor: {current_actor_lr:.6f}, Critic: {current_critic_lr:.6f}")
        self.memory.clear_memory()
        
    def is_ready_to_learn(self):
        return self.memory.is_ready()
    
    def compute_gae(self, rewards, values, dones):
        gae = torch.zeros_like(rewards)
        next_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * (gae[t + 1] if t + 1 < len(rewards) else 0)
        
        return gae