import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optimizer
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroResNet(nn.Module):
    def __init__(self, num_residual_blocks, num_channels=64):
        super(AlphaZeroResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_channels//2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels//2)
        self.conv2 = nn.Conv2d(in_channels=num_channels//2, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, num_channels) for _ in range(num_residual_blocks)]
        )
        
        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 3 * 3, 9)  # 9 possible moves in tic-tac-toe
        
        # Value head
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 3 * 3, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.residual_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 3 * 3)
        policy = F.softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 3 * 3)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class AlphaZeroModel:

    def __init__(self, config):
        self.config = config
        self.model = AlphaZeroResNet(
            config.num_blocks,
        )

        if not config.from_scratch:
            self.load(config.load_path)
            print("Loaded model from", config.load_path)
        if config.use_gpu:
            self.model = self.model.cuda()

        self.optimizer = optimizer.Adam(
            self.model.parameters(), weight_decay=config.l2_constant, lr=config.learning_rate
        )

    def predict(self, state):
        state = np.expand_dims(state, axis=0) # Add batch dimension
        state = torch.tensor(state, dtype=torch.float32)
        if self.config.use_gpu:
            state = state.cuda()
        with torch.no_grad():
            pi, v = self.model(state)
        return pi.cpu().numpy()[0], v.cpu().numpy()[0]

    def set_learning_rate(self, lr):
        """Sets the learning rate to the given value"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved to", path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
