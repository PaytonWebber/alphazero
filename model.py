import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optimizer
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):

        super(MLP, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for layer in self.fc_hidden:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroResNet(nn.Module):
    def __init__(self, num_residual_blocks, num_hidden_layers, num_channels=64):
        super(AlphaZeroResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_channels//2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels//2)
        self.conv2 = nn.Conv2d(in_channels=num_channels//2, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, num_channels) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.mlp_policy = MLP(num_channels*3*3, 256, 9, num_hidden_layers)
        self.softmax = nn.Softmax(dim=1)

        # Value head
        self.mlp_value = MLP(num_channels*3*3, 256, 1, num_hidden_layers)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.residual_blocks(x)
        x = x.view(-1, 64*3*3)

        # Policy head
        pi = self.mlp_policy(x)
        pi = self.softmax(pi)

        # Value head
        v = self.mlp_value(x)
        v = self.tanh(v)

        return pi, v


class AlphaZeroModel:

    def __init__(self, config):
        self.config = config
        self.model = AlphaZeroResNet(
            config.num_blocks,
            config.num_hidden_layers
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
