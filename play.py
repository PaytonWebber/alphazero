from tictactoe import TicTacToe
from mcts import MCTS_AlphaZero, Node_AlphaZero, MCTS_Basic, Node_Basic
from model import AlphaZeroModel
import data_structures as ds

winner_map = {0: "Player 0", 1: "Player 1", 2: "Draw"}

def play_mcts(user_player=0):
    state = TicTacToe()
    while not state.is_terminal():
        state.render()
        if state.current_player == user_player:
            while True:
                try:
                    x, y = map(int, input("Enter x y: ").split())
                    action = (x, y)
                    if action in state.legal_actions():
                        break
                    else:
                        print("Invalid Action")
                except KeyboardInterrupt:
                    print("\nExiting")
                    return
                except:
                    print("Invalid Input")
        else:
            root = Node_Basic(state, None, state.current_player)
            mcts = MCTS_Basic(root, 1.4, 100)
            action = mcts.search()
        state = state.step(action)
        print()

    state.render()
    print(f"Game Ended With Winner: {winner_map[state.winner()]}")

def play_alphazero(model, user_player=0):
    assert user_player in [0, 1]
    assert isinstance(model, AlphaZeroModel)
    state = TicTacToe()
    while not state.is_terminal():
        state.render()
        if state.current_player == user_player:
            while True:
                try:
                    x, y = map(int, input("Enter x y: ").split())
                    action = (x, y)
                    if action in state.legal_actions():
                        break
                    else:
                        print("Invalid Action")
                except KeyboardInterrupt:
                    print("\nExiting")
                    return
                except:
                    print("Invalid Input")
        else:
            root = Node_AlphaZero(state, None, state.current_player)
            mcts = MCTS_AlphaZero(root, model, 1.4, 100, False)
            action, _ = mcts.search()
        state = state.step(action)

    state.render()
    print(f"Game Ended With Winner: {winner_map[state.winner()]}")

if __name__ == "__main__":
    while True:
        opponent = input("Choose Your Opponent\n0: Basic MCTS\n1: AlphaZero Model\nChoice: ")
        if opponent in ["0", "1"]: break
        else: print("\nInvalid Input")

    if opponent == "0": play_mcts()
    else:
        net_config = ds.NetConfig()
        net_config.num_blocks = 20
        net_config.learning_rate = 0.002
        net_config.l2_constant = 1e-4
        net_config.from_scratch = True
        net_config.load_path = "models/model_107"
        net_config.use_gpu = True

        model = AlphaZeroModel(net_config)
        play_alphazero(model, 0)

