from model_code import ResNet_v4 as ResNet
from utils import TicTacToe
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_value_draw():
    model = ResNet()
    model.load_latest()
    model.cuda()
    model.eval()
    test_state = np.zeros((2, 3, 3))

    test_state[0] = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]])
    test_state[1] = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 0]])

    game = TicTacToe()
    game.board = test_state
    game.current_player = 1
    game.render()

    encoded_state = game._encode()
    state = torch.tensor(encoded_state, dtype=torch.float32)
    state = state.cuda().unsqueeze(0)
    pi, v = model.forward(state)
    pi = pi.to('cpu').detach().numpy().reshape(3, 3)
    print(pi)
    print(np.argmax(pi))
    assert v.item() == 0


def test_value_lose():
    model = ResNet()
    model.load_latest()
    model.cuda()
    model.eval()
    test_state = np.zeros((2, 3, 3))

    test_state[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_state[1] = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    game = TicTacToe()
    game.board = test_state
    game.current_player = 1
    game.render()

    encoded_state = game._encode()
    state = torch.tensor(encoded_state, dtype=torch.float32)
    state = state.cuda().unsqueeze(0)
    pi, v = model.forward(state)
    pi = pi.to('cpu').detach().numpy().reshape(3, 3)
    print(pi)
    print(np.argmax(pi))
    assert v.item() == -1


def test_value_win():
    model = ResNet()
    model.load_latest()
    model.cuda()
    model.eval()
    test_state = np.zeros((2, 3, 3))

    test_state[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    test_state[1] = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    game = TicTacToe()
    game.board = test_state
    game.current_player = 0
    game.render()

    encoded_state = game._encode()
    state = torch.tensor(encoded_state, dtype=torch.float32)
    state = state.cuda().unsqueeze(0)
    pi, v = model.forward(state)
    pi = pi.to('cpu').detach().numpy().reshape(3, 3)
    print(pi)
    print(np.argmax(pi))

    assert v.item() == 1
