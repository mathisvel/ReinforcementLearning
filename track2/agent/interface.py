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
TEAM_NAME = "Z-Team"

class ActorCriticNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCriticNet, self).__init__()
        # MLP for LunarLander (Input dim: 8)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        action_logits, _ = self.forward(x)
        # Deterministic action for evaluation (argmax)
        # Often yields better results on LunarLander than sampling
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
                                    
        # Load weights
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
        # No ImageWrapper for LunarLander
        self.wrapped_env = ArrayConversion(env=env, env_xp=np, target_xp=torch)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self.wrapped_env.step(action)