import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_dim=(84, 84, 1), action_dim=1):
        super(ActorCritic, self).__init__()
        
        # CNN for processing the game screen
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(input_dim)
        
        # Network for processing ball position and velocity
        self.ball_net = nn.Sequential(
            nn.Linear(4, 64),  # 4 = position(x,y) + velocity(vx,vy)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined features size
        combined_size = self.cnn_output_size + 64  # CNN features + ball features
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and std for continuous action
        self.action_mean = nn.Linear(128, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def _get_cnn_output_size(self, input_dim):
        """Calculate flattened size of CNN output."""
        x = torch.zeros(1, input_dim[2], input_dim[0], input_dim[1])
        return self.cnn(x).shape[1]
        
    def forward(self, observation):
        """Forward pass through the network."""
        # Process screen through CNN
        screen = observation['screen']
        if len(screen.shape) == 3:
            screen = screen.unsqueeze(0)
        screen = screen.permute(0, 3, 1, 2)  # NHWC to NCHW
        cnn_features = self.cnn(screen)
        
        # Process ball information
        ball_info = torch.cat([
            observation['ball_pos'],
            observation['ball_vel']
        ], dim=-1)
        ball_features = self.ball_net(ball_info)
        
        # Combine features
        combined_features = torch.cat([cnn_features, ball_features], dim=-1)
        
        # Get action distribution parameters
        actor_features = self.actor(combined_features)
        action_mean = self.action_mean(actor_features)
        action_std = torch.exp(self.action_log_std)
        
        # Get state value
        state_value = self.critic(combined_features)
        
        return action_mean, action_std, state_value
        
    def get_action(self, observation):
        """Sample action from the policy."""
        # Convert numpy arrays to tensors
        screen = torch.FloatTensor(observation['screen'])
        ball_pos = torch.FloatTensor(observation['ball_pos'])
        ball_vel = torch.FloatTensor(observation['ball_vel'])
        
        # Add batch dimension if needed
        if len(screen.shape) == 3:
            screen = screen.unsqueeze(0)
        if len(ball_pos.shape) == 1:
            ball_pos = ball_pos.unsqueeze(0)
        if len(ball_vel.shape) == 1:
            ball_vel = ball_vel.unsqueeze(0)
            
        observation = {
            'screen': screen,
            'ball_pos': ball_pos,
            'ball_vel': ball_vel
        }
        
        with torch.no_grad():
            action_mean, action_std, state_value = self(observation)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
        return action, action_log_prob, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, c1=1.0, c2=0.01):
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # PPO clipping parameter
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def update(self, rollouts):
        """Update policy using the currently gathered rollouts."""
        # Convert observations to tensors
        screens = torch.FloatTensor(rollouts['screens']).to(self.device)
        ball_positions = torch.FloatTensor(rollouts['ball_positions']).to(self.device)
        ball_velocities = torch.FloatTensor(rollouts['ball_velocities']).to(self.device)
        
        observations = {
            'screen': screens,
            'ball_pos': ball_positions,
            'ball_vel': ball_velocities
        }
        
        actions = torch.FloatTensor(rollouts['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollouts['returns']).to(self.device)
        advantages = torch.FloatTensor(rollouts['advantages']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for K epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(10):
            # Get current policy distributions and value predictions
            action_mean, action_std, values = self.actor_critic(observations)
            dist = Normal(action_mean, action_std)
            
            # Get current log probs and entropy
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate PPO policy loss
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            
        # Return average losses and entropy
        num_updates = 10
        return (
            total_policy_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates
        )
            
    def save(self, path):
        """Save model to disk."""
        torch.save(self.actor_critic.state_dict(), path)
        
    def load(self, path):
        """Load model from disk."""
        self.actor_critic.load_state_dict(torch.load(path))
