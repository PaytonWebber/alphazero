import torch
import torch.optim as optim
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



class ResNet_v3(nn.Module):
    def __init__(self, input_channels=2, output_dim=9, res_blocks=19, lr=0.002, weight_decay=1e-4):
        super(ResNet_v3, self).__init__()
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(64, 64) for _ in range(res_blocks)]
        )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            self.res_blocks
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(9, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(18, output_dim),
            nn.Softmax(dim=1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        x = self.model(x)
        v = self.value_head(x)
        pi = self.policy_head(x)
        return pi, v

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state = state.cuda().unsqueeze(0)
        pi, v = self.forward(state)
        return pi.cpu().detach().numpy(), v.cpu().detach().numpy()

    def save(self, epoch):
        print(f"Saving model at epoch {epoch}")
        torch.save(self.state_dict(), f"models/model_v3/model_{epoch}.pt")

    def load(self, epoch):
        print(f"Loading model at epoch {epoch}")
        self.load_state_dict(torch.load(f"models/model_v3/model_{epoch}.pt"))

    def save_latest(self):
        print("Saving latest model")
        torch.save(self.state_dict(), "models/model_v3/model_latest.pt")

    def load_latest(self):
        print("Loading latest model")
        self.load_state_dict(torch.load("models/model_v3/model_latest.pt"))


