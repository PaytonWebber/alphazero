from __future__ import annotations

from typing import Optional

import numpy as np


class Node_Basic:

    def __init__(
        self, state, action, to_play: int, parent: Optional[Node_Basic] = None
    ):
        self.state = state
        self.action = action
        self.to_play = to_play
        self.parent = parent
        self.reward_sum: float = 0.0
        self.N: int = 0  # Number of visits
        self.Q: float = 0.0  # Average reward
        self.children: list[Node_Basic] = list()

    def __str__(self) -> str:
        return f"{self.N} visits, {self.Q} average score, {self.action} action, {len(self.children)} children, To Play: {self.to_play}"

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (i.e. no children)."""
        return len(self.children) == 0

    def _exploitation_value(self) -> float:
        """The exploitation value of the node."""
        return self.Q

    def _exploration_value(self, C: float) -> float:
        """The exploration value of the node."""
        if self.parent is None:
            return 0  # For root node to make typechecker happy
        return C * safe_sqrt(np.log((self.parent.N + 1e-5)) / (self.N + 1e-5))

    def UCT(self, C: float) -> float:
        """Upper Confidence Bound for Trees (UCT) formula. Based on the UCB1 formula."""
        if self.parent is None:
            return 0  # For root node to make typechecker happy
        return self._exploitation_value() + self._exploration_value(C)


class MCTS_Basic:

    def __init__(self, root: Node_Basic, C: float, num_simulations: int):
        self.root = root
        self.C = C
        self.num_simulations = num_simulations

    def search(self) -> tuple:
        for _ in range(self.num_simulations):
            node = self.select()
            if not node.state.is_terminal() and node.is_leaf():
                self.expand(node)
                node = node.children[0]
            reward = self.playout(node)
            self.backpropagate(node, reward)
        return max(self.root.children, key=lambda n: n.N).action

    def select(self) -> Node_Basic:
        """Using the traditional UCT formula, select the best node to expand."""
        node = self.root
        while not node.is_leaf() and not node.state.is_terminal():
            node = max(node.children, key=lambda n: n.UCT(self.C))
        return node

    def expand(self, node: Node_Basic) -> None:
        """Expanding the node by adding a single, randomly selected child from the possible moves."""
        for action in node.state.legal_actions():
            new_state = node.state.step(action)
            new_node = Node_Basic(new_state, action, 1 - node.to_play, parent=node)
            node.children.append(new_node)

    def playout(self, node: Node_Basic) -> float:
        """Basic playout function. Randomly selects moves until the game is finished."""
        state = node.state
        while not state.is_terminal():
            action = state.actions[np.random.choice(len(state.actions))]
            state = state.step(action)
        return state.reward(node.to_play)

    def backpropagate(self, node: Node_Basic, reward: float) -> None:
        """Backpropagating the reward up the tree."""
        while node is not None:
            node.N += 1
            node.reward_sum += reward
            node.Q = node.reward_sum / node.N
            if node.parent is None:
                break
            node = node.parent
            reward = -reward


class Node_AlphaZero:

    def __init__(
        self,
        state,
        action,
        to_play: int,
        prior: float = 0.1,
        parent: Optional[Node_AlphaZero] = None,
    ):
        self.state = state
        self.action = action
        self.to_play = to_play
        self.parent = parent
        self.reward_sum: float = 0.0
        self.N: int = 0  # Number of visits
        self.Q: float = 0.0  # Average reward
        self.P: float = prior  # Prior probability
        self.children: list[Node_AlphaZero] = list()

    def __str__(self) -> str:
        return f"{self.N} visits, {self.P} prior, {self.Q} average score, {self.action} action, {len(self.children)} children, To Play: {self.to_play}"

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (i.e. no children)."""
        return len(self.children) == 0

    def _exploitation_value(self) -> float:
        """The exploitation value of the node."""
        return self.Q

    def _exploration_value(self, C: float) -> float:
        """The exploration value of the node."""
        if self.parent is None:
            return 0
        return self.P * C * safe_sqrt(self.parent.N) / (1 + self.N)

    def PUCT(self, C: float) -> float:
        """The PUCT formula."""
        if self.parent is None:
            return 0
        return self._exploitation_value() + self._exploration_value(C)


def safe_sqrt(x: float) -> float:
    return np.sqrt(x) if x > 0 else 0


class MCTS_AlphaZero:

    def __init__(
        self,
        root: Node_AlphaZero,
        model,
        C: float,
        num_simulations: int,
        training: bool = False,
    ) -> None:
        self.root = root
        self.model = model
        self.C = C
        self.num_simulations = num_simulations
        self.training = training

    def search(self) -> tuple:
        for _ in range(self.num_simulations):
            node = self.select()
            if not node.state.is_terminal() and node.is_leaf():
                self.expand(node, node == self.root)
                node = max(node.children, key=lambda n: n.PUCT(self.C))
            reward = self.playout(node)
            self.backpropagate(node, reward)
        policy = np.zeros((3, 3), dtype=np.float32)
        for child in self.root.children:
            policy[child.action] = child.N
        policy /= np.sum(policy)
        policy = policy.flatten()
        action = max(self.root.children, key=lambda n: n.N).action

        return action, policy

    def select(self) -> Node_AlphaZero:
        """Using the PUCT formula, select the best node to expand."""
        node = self.root
        while not node.is_leaf() and not node.state.is_terminal():
            node = max(node.children, key=lambda n: n.PUCT(self.C))
        return node

    def expand(self, node: Node_AlphaZero, add_noise: bool = False) -> None:
        """Expanding the node by adding a single, randomly selected child from the possible moves."""
        noise = np.random.dirichlet([0.03] * 9).reshape(3, 3)
        policy, _ = self.model.predict(node.state._encode())
        policy = np.reshape(policy, (3, 3))

        # Masking illegal actions
        new_policy = np.zeros((3, 3), dtype=np.float32)
        for action in node.state.legal_actions():
            new_policy[action] = policy[action]
            if add_noise and self.training:
                new_policy[action] = (1 - 0.25) * new_policy[action] + 0.25 * noise[
                    action
                ]
        normalized_policy = new_policy / np.sum(new_policy)

        for action in node.state.legal_actions():
            new_state = node.state.step(action)
            prior = normalized_policy[action].item()
            new_node = Node_AlphaZero(
                new_state, action, 1 - node.to_play, prior, parent=node
            )
            node.children.append(new_node)

    def playout(self, node: Node_AlphaZero) -> float:
        """Basic playout function. Randomly selects moves until the game is finished."""
        if node.state.is_terminal():
            reward = node.state.reward(node.state.current_player)
        else:
            _, value = self.model.predict(node.state._encode())
            reward = value
        return reward

    def backpropagate(self, node: Node_AlphaZero, reward: float) -> None:
        """Backpropagating the reward up the tree."""
        while node is not None:
            node.N += 1
            node.reward_sum += reward
            node.Q = node.reward_sum / node.N
            if node.parent is None:
                break
            node = node.parent
            reward = -reward
