import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network - simple but effective
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network - simple but effective
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state):
        return self.actor(state), self.critic(state)

class DartsPPO:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Initialize experience buffer
        self.experiences = deque(maxlen=500)  # Smaller buffer for faster adaptation
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        
        # Action parameters
        self.initial_std = 0.4
        self.min_std = 0.1
        self.std_decay = 0.995  # Slower decay
        self.action_std = self.initial_std
        
    def select_action(self, state, deterministic=False):
        """Select action using the current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_mean, _ = self.ac(state_tensor)
            action_mean = action_mean.cpu().numpy()[0]
        
        if deterministic:
            return action_mean, 0.0
        
        # Add exploration noise with decay
        self.action_std = max(self.min_std, self.action_std * self.std_decay)
        action = action_mean + np.random.normal(0, self.action_std, size=action_mean.shape)
        action = np.clip(action, -1, 1)
        
        # Calculate log probability
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
    
    def train(self, batch_size=32, epochs=10):
        """Train the agent using PPO."""
        if len(self.experiences) < batch_size:
            return
        
        # Convert experiences to numpy first
        states = np.array([exp['state'] for exp in self.experiences])
        actions = np.array([exp['action'] for exp in self.experiences])
        rewards = np.array([exp['reward'] for exp in self.experiences])
        next_states = np.array([exp['next_state'] for exp in self.experiences])
        old_log_probs = np.array([exp['log_prob'] for exp in self.experiences])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Train for multiple epochs
        for _ in range(epochs):
            # Get current action probabilities and state values
            action_mean, state_values = self.ac(states)
            
            # Calculate advantage
            with torch.no_grad():
                _, next_state_values = self.ac(next_states)
                advantages = rewards + self.gamma * next_state_values.squeeze() - state_values.squeeze()
            
            # Calculate new log probabilities
            action_std = torch.full_like(action_mean, self.action_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            
            # Calculate ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
            
            # Calculate losses with importance sampling
            weights = torch.ones_like(rewards)
            weights[rewards > rewards.mean()] = 2.0  # Give more weight to good experiences
            
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages) * weights
            actor_loss = actor_loss.mean()
            
            critic_loss = (advantages.pow(2) * weights).mean()
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()
    
    def save_model(self, path):
        """Save the model."""
        try:
            torch.save(self.ac.state_dict(), path)
        except Exception as e:
            print(f"Warning: Error saving model: {e}")
    
    def load_model(self, path):
        """Load the model."""
        try:
            self.ac.load_state_dict(torch.load(path))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model")
