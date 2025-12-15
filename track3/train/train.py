import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ArrayConversion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import os
import cv2

# --- NETWORK AND WRAPPER (IDENTICAL TO INTERFACE.PY) ---
# We copy them to ensure training uses exactly the same architecture

class ActorCriticNet(nn.Module):
    def __init__(self, action_dim):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        
        # Orthogonal Initialization (Vital for fast convergence)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor: small gain to start exploration
        nn.init.orthogonal_(self.actor.weight, 0.01)
        # Critic: standard gain
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.actor(x), self.critic(x)

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class ImageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.img_size = 84
        self.stack = deque(maxlen=4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, self.img_size, self.img_size), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process(obs)
        for _ in range(4): self.stack.append(processed)
        return np.array(self.stack), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process(obs)
        self.stack.append(processed)
        return np.array(self.stack), reward, terminated, truncated, info

    def _process(self, obs):
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

# --- TRAINING ALGO (PPO) ---

def make_env(env_name):
    def _init():
        env = gym.make(env_name, continuous=False)
        env = FrameSkip(env, skip=4)
        env = TimeLimit(env=env, max_episode_steps=1000)
        env = ImageWrapper(env)
        env = RecordEpisodeStatistics(env)
        return env
    return _init

def train():
    # 1. Config Environment
    env_name = 'CarRacing-v3'
    num_envs = 8
    # Vectorized Environment (Parallel)
    envs = gym.vector.AsyncVectorEnv([make_env(env_name) for _ in range(num_envs)])
    
    # 2. Config Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ActorCriticNet(action_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)
    
    # 3. PPO Hyperparameters
    num_iterations = 100
    steps_per_env = 256      # 256 * 8 = 2048 steps total per update
    batch_size = 256         # Larger batch size for stability
    gamma = 0.99
    lam = 0.95
    epochs = 4               # Reduced epochs for speed
    clip_eps = 0.2
    ent_coef = 0.01
    
    # Save path
    save_path = os.path.join(os.path.dirname(__file__), "../agent/weights.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 4. Training loop
    # Reset returns (obs, info)
    state, _ = envs.reset()
    state = torch.tensor(state).to(device)
    
    best_score = -np.inf
    global_step = 0
    
    print(f"Starting training {env_name} with {num_envs} envs...")
    
    for iteration in range(num_iterations):
        # --- A. Data Collection (Rollout) ---
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        for _ in range(steps_per_env):
            global_step += num_envs
            with torch.no_grad():
                logits, value = model(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Step Env (Vectorized)
            next_state_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Storage
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            dones.append(torch.tensor(done, dtype=torch.float32).to(device))
            
            state = torch.tensor(next_state_np).to(device)
            
            # Logging rewards from infos
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_rew = info["episode"]["r"]
                        if ep_rew > best_score:
                            best_score = ep_rew
                        # Simple print for progress
                        if np.random.rand() < 0.05: # Don't spam
                             print(f"Ep Reward: {ep_rew:.1f} | Best: {best_score:.1f}")

        # --- B. Advantage Calculation (GAE) ---
        with torch.no_grad():
            _, next_value = model(state)
            next_value = next_value.squeeze()
            
        returns = []
        gae = 0
        for i in reversed(range(steps_per_env)):
            # Mask (if done, next_val is not used for this time step)
            mask = 1.0 - dones[i].float()
            
            if i == steps_per_env - 1:
                next_val = next_value
            else:
                next_val = values[i+1]
            
            delta = rewards[i] + gamma * next_val * mask - values[i]
            gae = delta + gamma * lam * mask * gae
            returns.insert(0, gae + values[i])
            
        # Conversion & Flattening (Batch, Envs, ...) -> (Batch * Envs, ...)
        b_states = torch.stack(states).view(-1, 4, 84, 84)
        b_actions = torch.stack(actions).view(-1)
        b_log_probs = torch.stack(log_probs).view(-1)
        b_returns = torch.stack(returns).view(-1)
        b_values = torch.stack(values).view(-1)
        b_advantages = b_returns - b_values
        
        # Advantage Normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # --- C. Optimization (PPO Update) ---
        dataset_size = b_states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                idx = indices[start:start+batch_size]
                
                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_old_log_probs = b_log_probs[idx]
                mb_returns = b_returns[idx]
                mb_adv = b_advantages[idx]
                
                new_logits, new_values = model(mb_states)
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(new_values.squeeze(), mb_returns)
                
                loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
        # --- D. Logging & Save ---
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}/{num_iterations} | Global Steps: {global_step}")
            torch.save(model.state_dict(), save_path)
    
    envs.close()


if __name__ == "__main__":
    train()
