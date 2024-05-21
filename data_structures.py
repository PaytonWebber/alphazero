import numpy as np

class NetConfig:
    """Config class for AlphaZeroNet."""

    num_blocks: int
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

    def get(self):
        """Return the states, actions, and rewards of the entire game."""
        return self.states, self.policies, self.rewards


class ReplayBuffer:
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, game_history):
        """Add a self-play game to the buffer."""
        states, policies, values = game_history.get()
        for i in range(len(states)):
            if len(self.buffer) >= self.config.buffer_size:
                self.buffer.pop(0)
            self.buffer.append((states[i], policies[i], values[i]))

    def sample(self, batch_size):
        """Sample a batch of games from the buffer."""
        assert batch_size <= len(self.buffer)
        
        # Sample without replacement
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, policies, values = zip(*(self.buffer[idx] for idx in indices))
        
        return list(states), list(policies), list(values)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []

    def is_full(self):
        """Return True if the buffer is full."""
        return len(self.buffer) == self.config.buffer_size
