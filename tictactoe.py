from __future__ import annotations

from abc import abstractmethod

import numpy as np


class State:

    def __init__(self):
        self.current_player = 0
        self.board = None

    @abstractmethod
    def render(self) -> None:
        """Render the current state of the game."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        pass

    @abstractmethod
    def legal_actions(self) -> list:
        """Return a list of legal actions."""
        pass

    @abstractmethod
    def step(self, action) -> State:
        """Return the new state after taking the given action."""
        pass

    @abstractmethod
    def reward(self, player: int) -> int:
        """
        Return the reward for the given player of a terminal state.
        Note: This reward is the perspective of the other player.
        Return 1 if the other player wins, -1 if the other player loses, and 0 for a draw.
        """
        pass


winner_mask = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],  # Rows
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],  # Columns
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Diagonals
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
    ]
)


class TicTacToe(State):

    def __init__(
        self, board: np.ndarray = np.zeros((2, 3, 3)), current_player: int = 0
    ):
        super().__init__()
        self.board = board.reshape(2, 3, 3)
        self.current_player = current_player
        self.actions = self.legal_actions()

    def render(self) -> None:
        board = self.board.reshape(2, 9)
        print("Current Player:", self.current_player)
        for i in range(3):
            print("-------------")
            out = "| "
            for j in range(3):
                if board[0, i * 3 + j] == 1:
                    piece = "X"
                elif board[1, i * 3 + j] == 1:
                    piece = "O"
                else:
                    piece = " "
                out += piece + " | "
            print(out)
        print("-------------")

    def is_terminal(self) -> bool:
        return self.winner() != -1

    def legal_actions(self) -> list:
        board = self.board.reshape(2, 9)
        return [
            (i, j)
            for i in range(3)
            for j in range(3)
            if board[0, i * 3 + j] == 0 and board[1, i * 3 + j] == 0
        ]

    def step(self, action) -> TicTacToe:
        new_board = np.copy(self.board)
        new_board = new_board.reshape(2, 9)
        new_board[self.current_player, action[0] * 3 + action[1]] = 1
        return TicTacToe(new_board.reshape(2, 3, 3), 1 - self.current_player)

    def reward(self, player: int) -> int:
        if self.winner() == player:
            return -1
        if self.winner() == 1 - player:
            return 1
        return 0

    def winner(self) -> int:
        board = self.board.reshape(2, 9)
        for player in range(2):
            for mask in winner_mask:
                if np.all(board[player][mask == 1] == 1):
                    return player
        if np.all(board[0] + board[1] == 1):
            return 2
        return -1

    def _encode(self):
        """Encode the state into multiple channels for the neural network."""
        # First channel: current player controled cells
        # Second channel: opponent controled cells
        encoded = np.zeros((2, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.board[0, i, j] == 1: encoded[0, i, j] = 1
                elif self.board[1, i, j] == 1: encoded[1, i, j] = 1
        return encoded
        

