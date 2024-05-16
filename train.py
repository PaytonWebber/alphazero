import torch
from torch import nn
import numpy as np

from mcts import MCTS_AlphaZero, Node_AlphaZero
from tictactoe import TicTacToe
import data_structures as ds
from model import AlphaZeroModel


class Trainer:

    def __init__(self, net_config: ds.NetConfig, replay_config: ds.ReplayConfig, mcts_config: ds.MCTSConfig):
        self.model = AlphaZeroModel(net_config)
        self.replay_buffer = ds.ReplayBuffer(replay_config)
        self.mcts_config = mcts_config
        self.min_train_size = 512

    def self_play(self):
        game_history = ds.GameHistory()
        state = TicTacToe()
        while not state.is_terminal():
            root = Node_AlphaZero(state, None, state.current_player)
            mcts = MCTS_AlphaZero(root, self.model, self.mcts_config.C, self.mcts_config.num_simulations, self.mcts_config.training)
            action, pi = mcts.search()
            game_history.add(state._encode(), pi, state.current_player)
            state = state.step(action)
        game_history.update_reward(state.winner())
        self.replay_buffer.append(game_history)
        print(f"Game ended with winner: {state.winner()}")

    def train(self):
        print("Training...")
        for _ in range(10):
            # Sample a batch from the replay buffer
            batch_states, batch_policies, batch_values = self.replay_buffer.sample(self.replay_buffer.config.batch_size)

            # Convert lists of numpy arrays to numpy arrays and then to torch tensors
            batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32)
            batch_policies = torch.tensor(np.array(batch_policies), dtype=torch.float32)
            batch_values = torch.tensor(np.array(batch_values), dtype=torch.float32).view(-1, 1)  # Ensure correct shape

            # Send to GPU if available
            if self.model.config.use_gpu:
                batch_states = batch_states.cuda()
                batch_policies = batch_policies.cuda()
                batch_values = batch_values.cuda()

            # Zero gradients
            self.model.optimizer.zero_grad()

            # Forward pass
            policy_pred, value_pred = self.model.model(batch_states)

            # Calculate loss
            value_loss = nn.functional.mse_loss(value_pred, batch_values)
            policy_loss = -torch.sum(batch_policies * torch.log(policy_pred + 1e-8)) / batch_policies.size(0)
            loss = value_loss + policy_loss

            # Backward pass
            loss.backward()
            self.model.optimizer.step()

            # Calculate policy accuracy
            policy_pred_labels = torch.argmax(policy_pred, dim=1)
            policy_true_labels = torch.argmax(batch_policies, dim=1)
            policy_acc = (policy_pred_labels == policy_true_labels).float().mean().item()

            # Calculate value accuracy
            value_acc = (torch.abs(value_pred - batch_values) < 0.2).float().mean().item()  # Example threshold

            # Format to 4 decimal places
            value_loss = round(value_loss.item(), 4)
            policy_loss = round(policy_loss.item(), 4)
            value_acc = round(value_acc, 4)
            policy_acc = round(policy_acc, 4)

            print(f"Value loss: {value_loss}, Value acc: {value_acc}, Policy loss: {policy_loss}, Policy acc: {policy_acc}")


    def run(self):
        train_counter = 0
        try:
            while True:
                self.self_play()
                if len(self.replay_buffer) >= self.min_train_size:
                    self.train()
                    train_counter += 1
                    self.model.save(f"models/model_{train_counter}")
                    self.replay_buffer.clear()
                    
        except KeyboardInterrupt:
            print("Training stopped.")
            self.model.save(f"models/model_latest")


if __name__ == "__main__":
    # Network configuration
    net_config = ds.NetConfig()
    net_config.num_blocks = 5
    net_config.learning_rate = 0.002
    net_config.l2_constant = 1e-4
    net_config.from_scratch = True
    net_config.load_path = "models/model_latest"
    net_config.use_gpu = True

    # Replay buffer configuration
    replay_config = ds.ReplayConfig()
    replay_config.buffer_size = 640
    replay_config.batch_size = 64

    # MCTS configuration
    mcts_config = ds.MCTSConfig()
    mcts_config.num_simulations = 81
    mcts_config.C = 1.2
    mcts_config.training = True

    trainer = Trainer(net_config, replay_config, mcts_config)
    trainer.run()
