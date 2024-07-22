import os
import numpy as np
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
from torch.optim import Adam
import torch.nn.functional as F
import ray

from core import MCTS_az as MCTS, Node_az as Node, TicTacToe
from model_code import AlphaZeroNet

class GameHistory:
    """Class to store a game history."""
    def __init__(self):
        self.states = []
        self.policies = []
        self.rewards = []

    def __len__(self):
        return len(self.states)

    def add(self, state, policies, curr_player):
        """Add a state, action, and current player for the state."""
        self.states.append(state)
        self.policies.append(policies)
        self.rewards.append(curr_player)

    def update_reward(self, winning_player):
        """If the game did not end in a draw, update the rewards to reflect the winning player."""
        if winning_player == -1:
            return  # Draw
        for i in range(len(self.rewards)):
            if self.rewards[i] == winning_player:
                self.rewards[i] = 1
            else:
                self.rewards[i] = -1

    def get_history(self):
        return {
            'states': np.array(self.states),
            'policies': np.array(self.policies),
            'rewards': np.array(self.rewards)
        }


@ray.remote
class ModelActor:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def infer(self, state):
        with torch.no_grad():
            # Ensure state has 4 dimensions: [batch, channels, height, width]
            if state.ndim == 2:
                state = state[np.newaxis, np.newaxis, :, :]
            elif state.ndim == 3:
                state = state[np.newaxis, :, :, :]

            state = torch.tensor(state, dtype=torch.float32).clone().detach().to(self.device)

            pi, v = self.model(state)
            return pi.squeeze(0).cpu().numpy(), v.squeeze(0).cpu().numpy()

    def update_weights(self, state_dict: dict):
        self.model.load_state_dict(state_dict)

@ray.remote
def self_play(model_actor, C, num_simulations, seed, training=True):
    state = TicTacToe()

    np.random.seed(seed)
    game_history = GameHistory()
    while not state.is_terminal():
        root = Node(state, None, state.current_player)
        mcts = MCTS(root, model_actor, C, num_simulations, training)
        action, pi = mcts.search()
        game_history.add(state.encode(), pi, state.current_player)
        state = state.step(action)

    game_history.update_reward(state.winner())
    return game_history

def train_model(model, optimizer, states, policies, rewards):
    model.train()

    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32).cuda()

    policies = np.array(policies)
    policies = torch.tensor(policies, dtype=torch.float32).cuda()

    rewards = np.array(rewards)
    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).cuda()

    optimizer.zero_grad()
    pred_policies, pred_values = model(states)

    policy_loss = F.cross_entropy(pred_policies, policies)
    value_loss = F.mse_loss(pred_values, rewards)
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()
    print(f'Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}')
    return loss.item()

def train_loop(model_actor, model, optimizer, num_epochs, num_games, num_simulations, C, batch_size):
    for epoch in range(num_epochs):
        game_histories = ray.get([self_play.remote(model_actor, C, num_simulations, i) for i in range(num_games)])
        states, policies, rewards = [], [], []
        for game in game_histories:
            data = game.get_history()
            states.extend(data['states'])
            policies.extend(data['policies'])
            rewards.extend(data['rewards'])
        print(f'Collected {len(states)} samples')
        loss = train_model(model, optimizer, states, policies, rewards)
        print(f'Epoch {epoch} - Loss: {loss}')

        # Update model actor weights
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        model_actor.update_weights.remote(state_dict)

        if epoch % 10 == 0:
            print(f'Saving model at epoch {epoch}')
            # check if models folder exists
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), f'./models/model_{epoch}.pt')

if __name__ == '__main__':
    ray.init(address='auto')
    model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    model_actor = ModelActor.remote(AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9))  # Pass the model class, not an instance
    try:
        train_loop(model_actor, model, optimizer, num_epochs=100, num_games=20, num_simulations=50, C=1, batch_size=32)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), './models/model_latest.pt')
    ray.shutdown()
