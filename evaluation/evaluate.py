import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
from core import Node_az, MCTS_az, Node_basic, MCTS_basic, TicTacToe
from model_code import AlphaZeroNet

num_simulations = 100
C = 1.4

def evaluate(num_games: int = 100):

    model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9).cuda()
    model.load_state_dict(torch.load("./models/model_latest.pt"))

    wins = 0
    draws = 0
    losses = 0
    model_player = 0

    # play against basic MCTS
    try:
        for _ in range(num_games):
            state = TicTacToe()
            while not state.is_terminal():
                if state.current_player == model_player:
                    root = Node_az(state, state.current_player)
                    mcts = MCTS_az(root, model, C, num_simulations, training=False)
                    action, _ = mcts.search()
                else:
                    root = Node_basic(state, state.current_player)
                    mcts = MCTS_basic(root, num_simulations=num_simulations, C=C)
                    action = mcts.search()
                state = state.step(action)
            if state.winner() == model_player:
                print("Win")
                wins += 1
            elif state.winner() == 2:
                print("Draw")
                draws += 1
            else:
                print("Loss")
                losses += 1
            model_player = 1 - model_player
    except KeyboardInterrupt:
        pass

    print(f"\nWins: {wins}, Draws: {draws}, Losses: {losses}")

if __name__ == "__main__":
    evaluate(num_games=50)
