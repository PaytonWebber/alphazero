from mcts import MCTS_AlphaZero, Node_AlphaZero, MCTS_Basic, Node_Basic
from tictactoe import TicTacToe
from model_v3 import ResNet_v3 as ResNet

num_simulations = 25
C = 1.4

def evaluate(num_games: int = 100):

    model = ResNet()
    # model.load_latest()
    model.load(953)
    model.cuda()

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
                    root = Node_AlphaZero(state, None, state.current_player)
                    mcts = MCTS_AlphaZero(root, model, C, num_simulations, training=False)
                    action, _ = mcts.search()
                else:
                    root = Node_Basic(state, None, state.current_player)
                    mcts = MCTS_Basic(root, num_simulations=num_simulations, C=C)
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
