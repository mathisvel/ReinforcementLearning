import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os

# Add the parent directory to sys.path to import the agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent.interface import DQN

class ReplayBuffer:
    """
    A simple replay buffer to store transitions.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def soft_update(target_model, source_model, rho):
    """
    Soft update of target network parameters.
    θ_target = ρ * θ_source + (1 - ρ) * θ_target
    """
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(rho * param.data + (1.0 - rho) * target_param.data)

def train():
    """
    Main training loop for DQN agent on LunarLander-v3.
    """
    env = gym.make('LunarLander-v3')
    
    # Hyperparameters
    learning_rate = 5e-4
    gamma = 0.99            # Discount factor
    buffer_size = 100000    # Replay buffer size
    batch_size = 64         # Minibatch size
    rho = 0.005             # Polyak averaging coefficient for soft update
    epsilon_start = 1.0     # Initial exploration rate
    epsilon_end = 0.01      # Final exploration rate
    num_episodes = 600      # Total number of training episodes
    
    # Linear decay parameters
    epsilon_decay_episodes = 400
    epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_episodes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize networks
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
    # Copy weights from policy net to target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target net is only used for inference
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample() # Explore
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net.act(state_t).item() # Exploit
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            total_reward += reward
            
            # Train if buffer has enough samples
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # Double DQN Logic
                # 1. Select best action using policy net
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                    # 2. Evaluate that action using target net
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    # 3. Compute target Q-values
                    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
                
                # Get current Q-values from policy net
                current_q_values = policy_net(states).gather(1, actions)
                
                # Compute loss (MSE)
                loss = nn.MSELoss()(current_q_values, target_q_values)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Soft update target network
                soft_update(target_net, policy_net, rho)
        
        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon -= epsilon_step
            epsilon = max(epsilon_end, epsilon)
        
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            
    # Save trained weights
    save_path = os.path.join(os.path.dirname(__file__), '../agent/weights.pth')
    policy_net.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
