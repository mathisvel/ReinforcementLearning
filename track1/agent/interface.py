import gymnasium as gym
from gymnasium.wrappers import ArrayConversion
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

"""
    Please make sure this script does not import anything outside of
    - the standard library,
    - PyTorch, and
    - the packages listed in requirements.txt.
"""


# Specify your team name in a global variable like so:
TEAM_NAME = "Z-Team"



class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    A simple Multi-Layer Perceptron (MLP) with 2 hidden layers.
    """
    def __init__(self, input_size, output_size) -> None:
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),  # Input layer to first hidden layer
            nn.ReLU(),                   # Activation function
            nn.Linear(128, 128),         # First hidden layer to second hidden layer
            nn.ReLU(),                   # Activation function
            nn.Linear(128, output_size)  # Second hidden layer to output layer (Q-values)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        return self.net(x)

    @torch.no_grad()
    def act(self, x : torch.Tensor) -> torch.Tensor:
        """
        Select action with greedy policy (argmax of Q-values).
        """
        q_values = self.forward(x)
        actions = torch.argmax(q_values, dim=-1)
        return actions

    def save(self, path : str) -> None:
        """
        Save the model weights to a file.
        """
        torch.save(self.state_dict(), path)


class Policy:
    """
        Basic policy wrapper that enables loading and evaluating a pre-trained
        neural network policy.

        Modify the logic in the load and act functions to reflect your
        specific model architecture and other needs.

        Your agent will be evaluated by:
            a) Loading the approapriate model weights by calling the method load("agent/weights.pth")
            b) Interacting with the agent via act(observation) 
    """

    def __init__(self):
        self.model = None

    def load(self, model_path : str) -> None:
        """
        Construct and load pre-trained model from specified path.
        """
        lunar_action_count = 4
        lunar_obs_dim = 8
        
        # Initialize the model
        self.model = DQN(input_size=lunar_obs_dim,
                         output_size=lunar_action_count)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Set mode to eval (no gradient accumulation, etc.)
        self.model.eval()


    def act(self, observation : torch.Tensor) -> int:
        """
        Get action for a single observation from policy. 
        """
        if self.model is None:
            raise ValueError("Model was not loaded successfuly. Call load(path) before act().")

        # Get action from model
        action = self.model.act(observation)

        # Convert the single action to an integer for gym
        action = action.item()
        return action


class EnvironmentWrapper:
    """
        Basic environment wrapper that enables us to evaluate with the same
        environment transformations as you did during training.

        If you've used any wrappers that transform observations, actions,
        rewards and you want them to be applied during evaluation, you need to
        supply them here. See the example in the constructor.

        Your agent will interact with the evaluation environment via:
            a) Constructing the EnvWrapper with the env instance
            b) Applying wrappers you used in __init__
            c) Interacting with the env via reset() and step()

        You need to implement steps b) and c) as needed.

    """

    def __init__(self, env : gym.Env) -> None:
        """
        env - instance of a gym environment that will be wrapped
        
        Since we applied the ArrayConversion wrapper during training,
        we also need to apply it here to take effect during evaluation.
        This wrapper converts numpy arrays to torch tensors.
        """
        self.wrapped_env = ArrayConversion(env=env, env_xp=np, target_xp=torch)

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        """
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.
        """
        return self.wrapped_env.step(action)
