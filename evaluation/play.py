from tictactoe import TicTacToe
from mcts import MCTS_AlphaZero, Node_AlphaZero, MCTS_Basic, Node_Basic
from model_v3 import ResNet_v3 as ResNet

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
    assert isinstance(model, ResNet)
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
        model = ResNet()
        model = model.cuda()
        model.load_latest()
        play_alphazero(model, 0)

