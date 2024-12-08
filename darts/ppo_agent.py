import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state):
        return self.actor(state), self.critic(state)

class DartsPPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, action_std=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Initialize experience buffer
        self.experiences = deque(maxlen=1000)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        
        # Reduced action_std for less randomness and more stable learning
        self.action_std = action_std
        
    def select_action(self, state):
        """Select action using the current policy with exploration noise."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_mean, _ = self.ac(state_tensor)
            action_mean = action_mean.cpu().numpy()[0]
        
        # Add smaller exploration noise
        action = action_mean + np.random.normal(0, self.action_std, size=action_mean.shape)
        action = np.clip(action, -1, 1)
        
        # Calculate log probability (simplified)
        log_prob = -0.5 * np.sum(np.square((action - action_mean) / self.action_std))
        
        return action, log_prob
    
    def store_experience(self, state, action, reward, next_state, log_prob):
        """Store experience in buffer."""
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'log_prob': log_prob
        })
    
    def train(self, batch_size=32, epochs=5):
        """Train the agent using PPO."""
        if len(self.experiences) < batch_size:
            return
        
        # Sample batch
        batch_indices = np.random.choice(len(self.experiences), batch_size)
        batch = [self.experiences[i] for i in batch_indices]
        
        # Convert batch to tensors
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in batch]).to(self.device)
        
        for _ in range(epochs):
            action_mean, state_values = self.ac(states)
            
            with torch.no_grad():
                _, next_state_values = self.ac(next_states)
                advantages = rewards + self.gamma * next_state_values.squeeze() - state_values.squeeze()
            
            action_std = torch.full_like(action_mean, self.action_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
            
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = 0.5 * advantages.pow(2).mean()
            total_loss = actor_loss + critic_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def save_model(self, path):
        """Save the model."""
        torch.save(self.ac.state_dict(), path)
    
    def load_model(self, path):
        """Load the model."""
        self.ac.load_state_dict(torch.load(path))
