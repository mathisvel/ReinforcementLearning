import gymnasium as gym
from gymnasium.wrappers import ArrayConversion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple
import cv2

"""
    Track 3: Racecars from Pixels
    Minimalist and robust interface.
"""

# IMPORTANT: Put your team name here
TEAM_NAME = "Z-Team"

class ActorCriticNet(nn.Module):
    """
    CNN neural network to process CarRacing images.
    Input: (Batch, 4, 84, 84) -> 4 stacked grayscale frames.
    Output: Action logits (5 actions) and State value.
    """
    def __init__(self, action_dim):
        super(ActorCriticNet, self).__init__()
        
        # 1. Convolutional Layers (Feature Extraction)
        # Input: 4 channels (frames), 84x84
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # -> 32 x 20 x 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # -> 64 x 9 x 9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # -> 64 x 7 x 7
        
        self.flatten = nn.Flatten()
        
        # Size after convs: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 512)
        
        # 2. Heads (Actor & Critic)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Format handling (Batch or Single)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Pixel normalization 0-255 -> 0.0-1.0
        x = x.float() / 255.0
        
        # Convolutions + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten + Dense
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        
        return self.actor(x), self.critic(x)

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        """Used for evaluation (deterministic)"""
        action_logits, _ = self.forward(x)
        return torch.argmax(action_logits, dim=-1)

    def save(self, path : str) -> None:
        torch.save(self.state_dict(), path)


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
    """
    Essential preprocessing:
    1. RGB -> Grayscale
    2. Resize -> 84x84 (Standard DQN/PPO)
    3. Frame Stacking (4 frames)
    """
    def __init__(self, env):
        super().__init__(env)
        self.img_size = 84
        self.stack = deque(maxlen=4)
        
        # Modified observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, self.img_size, self.img_size), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process(obs)
        # Fill the buffer at the beginning
        for _ in range(4):
            self.stack.append(processed)
        return np.array(self.stack), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process(obs)
        self.stack.append(processed)
        return np.array(self.stack), reward, terminated, truncated, info

    def _process(self, obs):
        # Grayscale conversion (standard human perception weights)
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # Resizing
        return cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)


class Policy:
    """Wrapper required by the instructions for evaluation"""
    def __init__(self):
        self.model = None

    def load(self, model_path : str) -> None:
        # CarRacing has 5 discrete actions
        self.model = ActorCriticNet(action_dim=5)
        # Load on CPU to be safe
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def act(self, observation : torch.Tensor) -> int:
        if self.model is None:
            raise ValueError("Model not loaded!")
        return self.model.act(observation).item()


class EnvironmentWrapper:
    """Wrapper required by the instructions for evaluation"""
    def __init__(self, env : gym.Env) -> None:
        # 1. Frame Skip (Speeds up training and stabilizes)
        env = FrameSkip(env, skip=4)
        # 2. Image preprocessing
        env = ImageWrapper(env)
        # 3. Numpy -> Torch conversion (required by eval code)
        self.wrapped_env = ArrayConversion(env=env, env_xp=np, target_xp=torch)

    def reset(self, *args, **kwargs):
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action):
        return self.wrapped_env.step(action)

