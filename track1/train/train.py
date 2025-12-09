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
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def soft_update(target_model, source_model, rho):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(rho * param.data + (1.0 - rho) * target_param.data)

def train():
    env = gym.make('LunarLander-v3')
    
    # Hyperparameters
    learning_rate = 5e-4
    gamma = 0.99
    buffer_size = 100000
    batch_size = 64
    rho = 0.005 # Polyak averaging
    epsilon_start = 1.0
    epsilon_end = 0.01
    num_episodes = 600
    
    # Linear decay
    epsilon_decay_episodes = 400
    epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_episodes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net.act(state_t).item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # Double DQN
                # Select action using policy net
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                    # Evaluate action using target net
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
                
                current_q_values = policy_net(states).gather(1, actions)
                
                loss = nn.MSELoss()(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Soft update
                soft_update(target_net, policy_net, rho)
        
        # Linear decay
        if epsilon > epsilon_end:
            epsilon -= epsilon_step
            epsilon = max(epsilon_end, epsilon)
        
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            
    # Save weights
    save_path = os.path.join(os.path.dirname(__file__), '../agent/weights.pth')
    policy_net.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
