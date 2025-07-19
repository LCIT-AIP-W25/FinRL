import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))  # Output between -1 and 1
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # Initialize target networks
        self._update_target_networks(tau=1.0)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=100000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target network update rate
        self.batch_size = 64
        self.noise_std = 0.1  # Action noise for exploration
        
        # Training stats
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state, add_noise=True):
        """Select action with optional noise for exploration"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, next_actions)
        q_targets = rewards + (self.gamma * target_q_values * (1 - dones))
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, q_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
        self.critic_optimizer.step()
        
        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
    
    def _update_target_networks(self, tau=None):
        """Update target networks using soft update"""
        if tau is None:
            tau = self.tau
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
    
    def get_trading_signal(self, state, threshold=0.3):
        """Get trading signal based on current state"""
        action = self.select_action(state, add_noise=False)
        
        if action > threshold:
            return 'BUY', action[0]
        elif action < -threshold:
            return 'SELL', action[0]
        else:
            return 'HOLD', action[0]
