import gymnasium as gym
from gymnasium.wrappers import ArrayConversion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

"""
    Please make sure this script does not import anything outside of
    - the standard library,
    - PyTorch, and
    - the packages listed in requirements.txt.
"""

# Specify your team name in a global variable like so:
TEAM_NAME = "Haliloulou"

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
        # Gain sqrt(2) pour les couches avec Tanh
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        # Gain 0.01 pour l'acteur (maximise l'entropie au début)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        
        # Gain 1.0 pour le critique
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tanh est souvent meilleur pour les envs "physiques" (smoothness)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        action_logits, _ = self.forward(x)
        # Deterministic action for evaluation (argmax)
        actions = torch.argmax(action_logits, dim=-1)
        return actions

    def save(self, path : str) -> None:
        torch.save(self.state_dict(), path)

class Policy:
    """
        Policy wrapper for evaluation.
    """

    def __init__(self):
        self.model = None

    def load(self, model_path : str) -> None:
        """
            Construct and load pre-trained model from specified path
        """
        lunar_action_count = 4
        lunar_obs_dim = 8
        
        self.model = ActorCriticNet(input_size=lunar_obs_dim,
                                    output_size=lunar_action_count)
                                    
        # Chargement des poids
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Set mode to eval
        self.model.eval()

    def act(self, observation : torch.Tensor) -> int:
        """
            Get action for a single observation from policy. 
        """
        if self.model is None:
            raise ValueError("Model was not loaded successfuly. Call load(path) before act().")

        # Get action from model
        action = self.model.act(observation)

        # Convert to int
        return action.item()


class EnvironmentWrapper:
    """
        Wrappers applied during evaluation.
    """

    def __init__(self, env : gym.Env) -> None:
        # Pas d'ImageWrapper pour LunarLander
        self.wrapped_env = ArrayConversion(env=env, env_xp=np, target_xp=torch)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self.wrapped_env.step(action)
