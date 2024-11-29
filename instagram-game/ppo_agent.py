import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_dim=(84, 84, 1), action_dim=1):
        super(ActorCritic, self).__init__()
        
        # Enhanced CNN for processing the game screen
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Increased channels
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Added layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling
            nn.Flatten()
        )
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(input_dim)
        
        # Enhanced network for processing ball position and velocity
        self.ball_net = nn.Sequential(
            nn.Linear(4, 128),  # Increased width
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Added layer
            nn.ReLU()
        )
        
        # Combined features size
        combined_size = self.cnn_output_size + 128
        
        # Enhanced actor (policy) network
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
        # Learnable but bounded action std
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Enhanced critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def _get_cnn_output_size(self, input_dim):
        x = torch.zeros(1, input_dim[2], input_dim[0], input_dim[1])
        return self.cnn(x).shape[1]
        
    def forward(self, observation):
        # Process screen through CNN
        screen = observation['screen']
        if len(screen.shape) == 3:
            screen = screen.unsqueeze(0)
        screen = screen.permute(0, 3, 1, 2)
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
        # Bound the standard deviation
        action_std = torch.clamp(torch.exp(self.action_log_std), min=0.1, max=1.0)
        
        # Get state value
        state_value = self.critic(combined_features)
        
        return action_mean, action_std, state_value
        
    def get_action(self, observation):
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
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Use Adam optimizer with gradient clipping
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        
    def update(self, rollouts):
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
        
        # Increased number of epochs for better learning
        num_epochs = 15
        batch_size = 64  # Added mini-batch training
        num_batches = len(advantages) // batch_size
        
        for _ in range(num_epochs):
            # Random permutation for mini-batch training
            indices = torch.randperm(len(advantages))
            
            for start in range(0, len(advantages), batch_size):
                # Get mini-batch indices
                batch_indices = indices[start:start + batch_size]
                
                # Get batch observations
                batch_obs = {
                    'screen': observations['screen'][batch_indices],
                    'ball_pos': observations['ball_pos'][batch_indices],
                    'ball_vel': observations['ball_vel'][batch_indices]
                }
                
                # Get current policy distributions and value predictions
                action_mean, action_std, values = self.actor_critic(batch_obs)
                dist = Normal(action_mean, action_std)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(actions[batch_indices])
                entropy = dist.entropy().mean()
                
                # Calculate PPO policy loss
                ratio = torch.exp(curr_log_probs - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with clipping
                value_pred_clipped = values + torch.clamp(
                    values - values.detach(),
                    -self.epsilon, self.epsilon
                )
                value_losses = (values - returns[batch_indices]).pow(2)
                value_losses_clipped = (value_pred_clipped - returns[batch_indices]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
            
        # Return average losses and entropy
        num_updates = num_epochs * num_batches
        return (
            total_policy_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates
        )
            
    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
