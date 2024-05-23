import numpy as np
from torch.utils.data import Dataset

class NetConfig:
    """Config class for AlphaZeroNet."""

    num_blocks: int
    num_hidden_layers: int
    learning_rate: float
    l2_constant: float
    from_scratch: bool
    load_path: str
    use_gpu: bool


class ReplayConfig:
    """Config class for ReplayBuffer."""

    buffer_size: int
    batch_size: int

class MCTSConfig:
    """Config class for MCTS."""

    num_simulations: int
    C: float
    training: bool

class GameHistory:
    """Class to store a game history."""

    def __init__(self):
        self.states = []
        self.policies = []
        self.rewards = []

    def __len__(self):
        return len(self.states)

    def add(self, state, policies, curr_player):
        """Add a state, action, and current player for the the state."""
        self.states.append(state)
        self.policies.append(policies)
        self.rewards.append(curr_player)

    def update_reward(self, winning_player):
        """If the game did not end in a draw, update the rewards to reflect the winning player."""
        if winning_player == -1: return # Draw
        for i in range(len(self.rewards)):
            if self.rewards[i] == winning_player:
                self.rewards[i] = 1
            else:
                self.rewards[i] = -1

        self._augment_history()

    def _augment_history(self):
        """Augeument the history by rotating the board state and policies."""
        augmented_states = []
        augmented_policies = []
        augmented_rewards = []
        for i in range(len(self.states)):
            state = self.states[i]
            policies = self.policies[i]
            reward = self.rewards[i]

            # Original state policy and reward
            augmented_states.append(state)
            augmented_policies.append(policies)
            augmented_rewards.append(reward)

            # Rotate the board state and policies
            for _ in range(4):
                state = np.rot90(state)
                policies = np.rot90(policies.reshape(3, 3)).flatten()
                augmented_states.append(state)
                augmented_policies.append(policies)
                augmented_rewards.append(reward)

            # Flip the board state and policies
            state = np.flipud(state)
            policies = np.flipud(policies.reshape(3, 3)).flatten()
            augmented_states.append(state)
            augmented_policies.append(policies)
            augmented_rewards.append(reward)

        self.states = augmented_states
        self.policies = augmented_policies
        self.rewards = augmented_rewards
        
    def get(self):
        """Return the states, actions, and rewards of the entire game."""
        return self.states, self.policies, self.rewards


class ReplayBuffer(Dataset):
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return {
                "states": self.buffer[idx][0],
                "policies": self.buffer[idx][1],
                "rewards": self.buffer[idx][2]
                }

    def append(self, game_history):
        """Add a self-play game to the buffer."""
        states, policies, values = game_history.get()
        for i in range(len(states)):
            if len(self.buffer) >= self.config.buffer_size:
                self.buffer.pop(0)
            self.buffer.append((states[i], policies[i], values[i]))

    def clear(self):
        """Clear the buffer."""
        self.buffer = []

    def is_full(self):
        """Return True if the buffer is full."""
        return len(self.buffer) == self.config.buffer_size
