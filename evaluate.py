from mcts import MCTS_AlphaZero, Node_AlphaZero, MCTS_Basic, Node_Basic
from tictactoe import TicTacToe

import data_structures as ds
from model import AlphaZeroModel

def evaluate(num_games: int = 100):
    net_config = ds.NetConfig()
    net_config.num_blocks = 5
    net_config.learning_rate = 0.002
    net_config.l2_constant = 1e-4
    net_config.from_scratch = True
    net_config.load_path = "models/model_latest"
    net_config.use_gpu = True


    model = AlphaZeroModel(net_config)

    wins = 0
    draws = 0
    losses = 0

    model_player = 0
    # play against basic MCTS
    for _ in range(num_games):
        state = TicTacToe()
        while not state.is_terminal():
            if state.current_player == model_player:
                root = Node_AlphaZero(state, None, state.current_player)
                mcts = MCTS_AlphaZero(root, model, 1.0, 100, False)
                action, _ = mcts.search()
            else:
                root = Node_Basic(state, None, state.current_player)
                mcts = MCTS_Basic(root, num_simulations=100, C=1.0)
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

    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

if __name__ == "__main__":
    evaluate(num_games=20)
