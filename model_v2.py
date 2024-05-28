import torch
import torch.nn as nn

args = {
    'M': 3,
    'N': 3,
    'dropout': 0.1
}

class SmallBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallBlock, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
        )
    def forward(self, X):
        return self.model(X)
        
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ResnetBlock, self).__init__()
        self.model = nn.Sequential(
            SmallBlock(in_channels, mid_channels),
            nn.ReLU(),
            SmallBlock(mid_channels, in_channels)
        )
    def forward(self, x):
        input = x
        x = self.model(x)
        x += input
        x = nn.ReLU()(x)
        return x
    
class DropoutBlock(nn.Module):
    def __init__(self, in_units, out_units):
        super(DropoutBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.LayerNorm(out_units),
            nn.ReLU(),
            nn.Dropout(p=args['dropout'])
        )
    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, in_units, out_units):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.LayerNorm(out_units),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class ResNet_v2(nn.Module):
    def __init__(self, H=[200,100], num_channels=64):
        # input shape: batch_size x 2 x args.M x args.N
        super(ResNet_v2, self).__init__()
        
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.middle_blocks = nn.Sequential(
            *[ResnetBlock(num_channels,num_channels) for _ in range(5)]
        )
        self.mlp = nn.Sequential(
            *[MLP(num_channels * args['M'] * args['N'], num_channels * args['M'] * args['N']) for _ in range(2)]
        )
        self.model = nn.Sequential(
            self.initial_block,
            self.middle_blocks,
            nn.Flatten(start_dim=1),
            self.mlp,
            nn.Linear(num_channels*args['M']*args['N'], H[0]),
            nn.LayerNorm(H[0]),
            nn.ReLU(),
            nn.Linear(H[0], H[1]),
            nn.LayerNorm(H[1]),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(H[1], H[1]),
            nn.ReLU(),
            nn.Linear(H[1], 1),
            nn.Tanh()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(H[1], H[1]),
            nn.ReLU(),
            nn.Linear(H[1], args['M']*args['N']),
            nn.Softmax(dim=1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=1e-4)
    
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
        torch.save(self.state_dict(), f"models/model_v2/model_{epoch}.pt")

    def load(self, epoch):
        print(f"Loading model at epoch {epoch}")
        self.load_state_dict(torch.load(f"models/model_v2/model_{epoch}.pt"))

    def save_latest(self):
        print("Saving latest model")
        torch.save(self.state_dict(), "models/model_v2/model_latest.pt")

    def load_latest(self):
        print("Loading latest model")
        self.load_state_dict(torch.load("models/model_v2/model_latest.pt"))


