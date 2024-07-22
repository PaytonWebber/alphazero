import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from mcts import MCTS_AlphaZero, Node_AlphaZero
from utils import TicTacToe
from utils import data_structures as ds
from model_code import ResNet_v4


class Trainer:
    def __init__(self, mcts_config: ds.MCTSConfig, load_latest=False, load_epoch=None, global_step=0):
        self.model = ResNet_v4(lr=0.002)
        if load_latest:
            self.model.load_latest()
        elif load_epoch:
            self.model.load(load_epoch)
        self.model.cuda()
        self.writer = SummaryWriter('logs/model_v4')
        self.replay_buffer = ds.ReplayBuffer(10000)
        self.mcts_config = mcts_config
        self.min_train_size = 1024
        self.batch_size = 64
        self.global_step = global_step

    def self_play(self):
        game_history = ds.GameHistory()
        state = TicTacToe()
        while not state.is_terminal():
            root = Node_AlphaZero(state, None, state.current_player)
            mcts = MCTS_AlphaZero(root, self.model, self.mcts_config.C,
                                  self.mcts_config.num_simulations, self.mcts_config.training)
            action, pi = mcts.search()
            game_history.add(state._encode(), pi, state.current_player)
            state = state.step(action)
        game_history.update_reward(state.winner())
        self.replay_buffer.append(game_history)
        print(f"Game ended with winner: {state.winner()}")

    @staticmethod
    def custom_collate_fn(batch):
        states = [item['states'] for item in batch]
        policies = [item['policies'] for item in batch]
        rewards = [item['rewards'] for item in batch]

        states = np.stack(states)
        policies = np.stack(policies)
        rewards = np.stack(rewards)

        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'policies': torch.tensor(policies, dtype=torch.float32),
            'rewards': torch.tensor(rewards, dtype=torch.float32)
        }

    def train(self):
        dataloader = DataLoader(self.replay_buffer, batch_size=self.batch_size,
                                shuffle=True, drop_last=True, collate_fn=self.custom_collate_fn)
        self.model.train()

        for epoch in range(2):
            for batch in dataloader:
                states = batch['states'].cuda()
                policies = batch['policies'].cuda()
                rewards = batch['rewards'].cuda()

                self.model.optimizer.zero_grad()
                pi, v = self.model(states)
                policy_loss = torch.nn.CrossEntropyLoss()(pi, policies)
                value_loss = torch.nn.MSELoss()(v, rewards.view(-1, 1))
                loss = policy_loss + value_loss
                loss.backward()
                self.model.optimizer.step()

                self.writer.add_scalar(
                    'Loss/policy', policy_loss.item(), self.global_step)
                self.writer.add_scalar(
                    'Loss/value', value_loss.item(), self.global_step)

                pi_pred = torch.argmax(pi, dim=1)
                pi_acc = torch.sum(pi_pred == torch.argmax(
                    policies, dim=1)).item() / self.batch_size

                v_acc = torch.sum(torch.round(
                    v) == rewards.view(-1, 1)).item() / self.batch_size

                print(
                    f"Epoch: {epoch}, Pi Loss: {policy_loss.item():.4f}, V Loss: {value_loss.item():.4f}, Pi Acc: {pi_acc:.4f}, V Acc: {v_acc:.4f}")

                self.global_step += 1

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)
                self.writer.add_histogram(f'{name}.grad', param.grad, epoch)

    def run(self, train_counter: int = 0):
        try:
            while True:
                self.self_play()
                if self.replay_buffer.is_full() or len(self.replay_buffer) >= self.min_train_size:
                    self.train()
                    train_counter += 1
                    self.model.save(train_counter)
                    self.model.save_latest()
                    self.replay_buffer.clear()

        except KeyboardInterrupt:
            print("Training stopped.")
            self.model.save_latest()
            self.writer.close()
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        #     self.model.save_latest()
        #     self.writer.close()


if __name__ == "__main__":
    mcts_config = ds.MCTSConfig()
    mcts_config.num_simulations = 150
    mcts_config.C = 1.4
    mcts_config.training = True

    trainer = Trainer(mcts_config, load_latest=False,
                      load_epoch=None, global_step=0)
    trainer.run(train_counter=0)
