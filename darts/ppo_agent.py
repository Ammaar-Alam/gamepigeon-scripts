import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import json
import os

class DartsNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DartsNet, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(128, action_dim)
        self.policy_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value = nn.Linear(128, 1)
    
    def forward(self, state):
        features = self.shared(state)
        
        # Policy
        action_mean = self.policy_mean(features)
        action_std = F.softplus(self.policy_std).expand_as(action_mean)
        
        # Value
        value = self.value(features)
        
        return action_mean, action_std, value

class DartsPPO:
    def __init__(self, state_dim=6, action_dim=5, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network
        self.network = DartsNet(state_dim, action_dim).to(self.device)
        if load_path and os.path.exists(load_path):
            self.network.load_state_dict(torch.load(load_path))
            print(f"Loaded model from {load_path}")
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.learning_rate = 3e-4
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Experience buffer
        self.buffer_size = 1000
        self.experiences = deque(maxlen=self.buffer_size)
        
        # Normalize state and actions
        self.state_mean = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        self.action_mean = torch.zeros(action_dim)
        self.action_std = torch.ones(action_dim)
        
        # Load normalization stats if available
        self.load_normalization_stats()
    
    def normalize_state(self, state):
        """Normalize state using running statistics."""
        state_tensor = torch.FloatTensor(state)
        return (state_tensor - self.state_mean) / (self.state_std + 1e-8)
    
    def normalize_action(self, action):
        """Normalize action using running statistics."""
        action_tensor = torch.FloatTensor(action)
        return (action_tensor - self.action_mean) / (self.action_std + 1e-8)
    
    def denormalize_action(self, normalized_action):
        """Convert normalized action back to original space."""
        return normalized_action * self.action_std + self.action_mean
    
    def update_normalization_stats(self, states, actions):
        """Update running statistics for normalization."""
        # Update state statistics
        states_tensor = torch.FloatTensor(states)
        self.state_mean = 0.99 * self.state_mean + 0.01 * states_tensor.mean(dim=0)
        self.state_std = 0.99 * self.state_std + 0.01 * states_tensor.std(dim=0)
        
        # Update action statistics
        actions_tensor = torch.FloatTensor(actions)
        self.action_mean = 0.99 * self.action_mean + 0.01 * actions_tensor.mean(dim=0)
        self.action_std = 0.99 * self.action_std + 0.01 * actions_tensor.std(dim=0)
    
    def save_normalization_stats(self):
        """Save normalization statistics to file."""
        stats = {
            'state_mean': self.state_mean.tolist(),
            'state_std': self.state_std.tolist(),
            'action_mean': self.action_mean.tolist(),
            'action_std': self.action_std.tolist()
        }
        with open('models/normalization_stats.json', 'w') as f:
            json.dump(stats, f)
    
    def load_normalization_stats(self):
        """Load normalization statistics from file."""
        try:
            with open('models/normalization_stats.json', 'r') as f:
                stats = json.load(f)
                self.state_mean = torch.FloatTensor(stats['state_mean'])
                self.state_std = torch.FloatTensor(stats['state_std'])
                self.action_mean = torch.FloatTensor(stats['action_mean'])
                self.action_std = torch.FloatTensor(stats['action_std'])
        except FileNotFoundError:
            pass
    
    def select_action(self, state):
        """Select action using current policy."""
        with torch.no_grad():
            state = self.normalize_state(state).to(self.device)
            action_mean, action_std, _ = self.network(state.unsqueeze(0))
            
            # Sample action from normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Denormalize action
            action = self.denormalize_action(action)
            
            return action.cpu().numpy()[0], log_prob.cpu().numpy()
    
    def store_experience(self, state, action, reward, next_state, log_prob):
        """Store experience in buffer."""
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'log_prob': log_prob
        })
    
    def train(self, batch_size=32, epochs=10):
        """Train the PPO model on stored experiences."""
        if len(self.experiences) < batch_size:
            return
        
        # Convert experiences to tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        old_log_probs = []
        
        # Sample random batch
        batch = random.sample(self.experiences, batch_size)
        for exp in batch:
            states.append(exp['state'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            next_states.append(exp['next_state'])
            old_log_probs.append(exp['log_prob'])
        
        # Update normalization statistics
        self.update_normalization_stats(states, actions)
        
        # Convert to tensors and normalize
        states = torch.FloatTensor([self.normalize_state(s) for s in states]).to(self.device)
        actions = torch.FloatTensor([self.normalize_action(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor([self.normalize_state(s) for s in next_states]).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for _ in range(epochs):
            # Get current policy and value predictions
            action_mean, action_std, values = self.network(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio and surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * rewards
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), rewards)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save_model(self, path):
        """Save model to file."""
        torch.save(self.network.state_dict(), path)
        self.save_normalization_stats()
        print(f"Model saved to {path}")
