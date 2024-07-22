from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m: nn.Module):
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(m, nn.Module)

    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):

    def __init__(self, num_filters: int):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self,
                 input_shape: Tuple,
                 num_actions: int,
                 num_blocks: int = 19,
                 num_filters: int = 256,
                 num_hidden_units: int = 256):
        super(AlphaZeroNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = input_shape

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_block = nn.Sequential(
            nn.Conv2d(c, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResNetBlock(num_filters) for _ in range(num_blocks)]
        )

        conv_out = self.get_conv_output(input_shape)

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out // 128, num_actions),  # Divide by 128 because we reduced channels from 256 to 2
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out // 256, num_hidden_units),  # Divide by 256 because we reduced channels from 256 to 1
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1),
            nn.Tanh()
        )

        self.apply(init_weights)

    def conv_output_size(self, size, kernel_size=3, stride=1, padding=1):
        return (size + 2 * padding - kernel_size) // stride + 1

    def get_conv_output(self, shape):
        o = self.conv_output_size(shape[1])
        return int(o * o * 256)  # 256 is the number of filters
    
    def infer(self, state):
        with torch.no_grad():
            # Ensure state has 4 dimensions: [batch, channels, height, width]
            if state.ndim == 2:
                state = state[np.newaxis, np.newaxis, :, :]
            elif state.ndim == 3:
                state = state[np.newaxis, :, :, :]

            state = torch.tensor(state, dtype=torch.float32).clone().detach().to(self.device)

            pi, v = self.forward(state)
            return pi.squeeze(0).cpu().numpy(), v.squeeze(0).cpu().numpy()

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
