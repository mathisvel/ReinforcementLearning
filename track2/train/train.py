import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, ArrayConversion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os

# --- CLASSES DUPLIQUÉES DE INTERFACE.PY ---

class ActorCriticNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCriticNet, self).__init__()
        # MLP optimisé avec Tanh et Init Orthogonale
        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 128)
        
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)

        # Initialisation des poids (Tricks PPO)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x : torch.Tensor):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

# --- PPO TRAINING LOGIC ---

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Calcule Generalized Advantage Estimation (GAE)"""
    gae = 0
    returns = []
    values = values + [next_value]
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
        
    return returns

def ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_eps=0.2, epochs=10, batch_size=64, ent_coef=0.01):
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            
            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_log_probs_old = log_probs_old[idx]
            mb_returns = returns[idx]
            mb_advantages = advantages[idx]
            
            # Forward pass
            logits, values = model(mb_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            
            # Ratio
            ratio = torch.exp(new_log_probs - mb_log_probs_old)
            
            # Surrogate Loss
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss
            critic_loss = F.mse_loss(values.squeeze(), mb_returns)
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()

def train(env, model, optimizer, num_iterations=1000):
    # Hyperparams PPO optimisés
    steps_per_iter = 2048
    gamma = 0.99
    lam = 0.95
    epochs = 10            # Augmenté pour mieux apprendre sur chaque batch
    batch_size = 64
    
    # Annealing
    initial_lr = 3e-4
    initial_ent_coef = 0.01
    
    model.train()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "../agent/weights.pth")
    
    state, _ = env.reset()
    
    best_avg_reward = -np.inf
    
    for iteration in range(num_iterations):
        # Linear Annealing
        frac = 1.0 - (iteration / num_iterations)
        lr_now = initial_lr * frac
        ent_coef_now = initial_ent_coef * frac
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
            
        # Buffers
        log_probs = []
        values = []
        rewards = []
        dones = []
        states = []
        actions = []
        
        # 1. Collection de données
        for _ in range(steps_per_iter):
            with torch.no_grad():
                logits, value = model(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            
            if done:
                state, _ = env.reset()
        
        # 2. Bootstrapping
        with torch.no_grad():
            _, next_value = model(state)
            next_value = next_value.item()
            
        # 3. Calcul des avantages
        returns = compute_gae(rewards, values, next_value, dones, gamma, lam)
        
        batch_states = torch.stack(states)
        batch_actions = torch.stack(actions)
        batch_log_probs = torch.stack(log_probs)
        batch_returns = torch.tensor(returns, dtype=torch.float32)
        batch_values = torch.tensor(values, dtype=torch.float32)
        
        advantages = batch_returns - batch_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 4. Update
        ppo_update(model, optimizer, batch_states, batch_actions, batch_log_probs, 
                   batch_returns, advantages, epochs=epochs, batch_size=batch_size, ent_coef=ent_coef_now)
        
        # Logging
        ep_rewards = []
        current_ep_rew = 0
        for r, d in zip(rewards, dones):
            current_ep_rew += r
            if d:
                ep_rewards.append(current_ep_rew)
                current_ep_rew = 0
        
        if len(ep_rewards) > 0:
            avg_reward = np.mean(ep_rewards)
        else:
            avg_reward = sum(rewards)
            
        if (iteration + 1) % 1 == 0:
            print(f"Iter {iteration+1}/{num_iterations}, Avg Reward: {avg_reward:.2f}, LR: {lr_now:.2e}")
            
            # Sauvegarde uniquement si on bat le record (et > 150 pour éviter le bruit du début)
            if avg_reward > best_avg_reward and avg_reward > 150:
                best_avg_reward = avg_reward
                torch.save(model.state_dict(), save_path)
                print(f"  >>> Nouveau record ! Sauvegarde ({avg_reward:.2f})")
            elif (iteration + 1) % 20 == 0:
                 torch.save(model.state_dict(), save_path) # Sauvegarde de sécurité

if __name__ == "__main__":
    env_name = 'LunarLander-v3'
    env = gym.make(env_name)
    
    # Wrappers
    env = TimeLimit(env=env, max_episode_steps=500)
    env = ArrayConversion(env=env, env_xp=np, target_xp=torch)
    env = RecordEpisodeStatistics(env=env, stats_key='episode', buffer_length=10)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCriticNet(obs_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5) # Adam Epsilon optimisé

    print(f"Début de l'entraînement PPO EXPERT sur {env_name}...")
    train(env, model, optimizer, num_iterations=1000)
    
    env.close()
