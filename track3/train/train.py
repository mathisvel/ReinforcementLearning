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

# --- RÉSEAU ET WRAPPER (IDENTIQUES À INTERFACE.PY) ---
# On les copie pour garantir que l'entraînement utilise exactement la même architecture

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
        
        # Initialisation Orthogonale (Vital pour la convergence rapide)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Acteur: petit gain pour démarrer l'exploration
        nn.init.orthogonal_(self.actor.weight, 0.01)
        # Critique: gain standard
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

# --- ALGO D'ENTRAÎNEMENT (PPO) ---

def train():
    # 1. Config Environment
    env_name = 'CarRacing-v3'
    env = gym.make(env_name, continuous=False) # Actions discrètes
    env = TimeLimit(env=env, max_episode_steps=1000)
    env = ImageWrapper(env) # Notre preprocessing
    env = ArrayConversion(env=env, env_xp=np, target_xp=torch)
    env = RecordEpisodeStatistics(env=env, buffer_length=10)
    
    # 2. Config Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ActorCriticNet(action_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5) # LR standard PPO
    
    # 3. Hyperparamètres PPO
    num_iterations = 2000    # Nombre total de mises à jour
    steps_per_iter = 2048    # Données collectées par mise à jour
    batch_size = 64
    gamma = 0.99
    lam = 0.95               # GAE lambda
    epochs = 10              # Tours d'optimisation par itération
    clip_eps = 0.2
    ent_coef = 0.01          # Exploration
    
    # Path sauvegarde
    save_path = os.path.join(os.path.dirname(__file__), "../agent/weights.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 4. Boucle d'entraînement
    state, _ = env.reset()
    state = torch.tensor(state).to(device) # Conversion initiale
    
    best_score = -np.inf
    
    print(f"Démarrage training {env_name}...")
    
    for iteration in range(num_iterations):
        # --- A. Collection de données (Rollout) ---
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        for _ in range(steps_per_iter):
            with torch.no_grad():
                logits, value = model(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Step Env
            next_state_np, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Stockage
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            
            state = torch.tensor(next_state_np).to(device)
            
            if done:
                state_np, _ = env.reset()
                state = torch.tensor(state_np).to(device)
                
        # --- B. Calcul des Avantages (GAE) ---
        with torch.no_grad():
            _, next_value = model(state)
            
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            # Masque pour les terminaux
            mask = 1.0 - float(dones[i])
            next_val = values[i+1].item() if i + 1 < len(values) else next_value.item()
            
            delta = rewards[i] + gamma * next_val * mask - values[i].item()
            gae = delta + gamma * lam * mask * gae
            returns.insert(0, gae + values[i].item())
            
        # Conversion Batch
        b_states = torch.stack(states)
        b_actions = torch.stack(actions)
        b_log_probs = torch.stack(log_probs)
        b_returns = torch.tensor(returns, device=device)
        b_values = torch.stack(values).squeeze()
        b_advantages = b_returns - b_values
        
        # Normalisation Avantages (Crucial pour PPO)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # --- C. Optimisation (PPO Update) ---
        dataset_size = b_states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                idx = indices[start:start+batch_size]
                
                # Mini-batch
                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_old_log_probs = b_log_probs[idx]
                mb_returns = b_returns[idx]
                mb_adv = b_advantages[idx]
                
                # Forward
                new_logits, new_values = model(mb_states)
                dist = Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                critic_loss = F.mse_loss(new_values.squeeze(), mb_returns)
                
                # Total Loss
                loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient Clipping
                optimizer.step()
        
        # --- D. Logging & Save ---
        avg_rew = sum(rewards) / (dones.count(True) + 1e-9)
        if (iteration + 1) % 1 == 0:
            print(f"Iter {iteration+1}/{num_iterations} | Avg Reward: {avg_rew:.2f} | Best: {best_score:.2f}")
            
            # Sauvegarde meilleur modèle (si score décent)
            if avg_rew > best_score and avg_rew > 50:
                best_score = avg_rew
                torch.save(model.state_dict(), save_path)
                print(f"  >>> New Best Model Saved!")
            # Sauvegarde périodique (au cas où)
            elif (iteration + 1) % 50 == 0:
                torch.save(model.state_dict(), save_path)
    
    env.close()

if __name__ == "__main__":
    train()
