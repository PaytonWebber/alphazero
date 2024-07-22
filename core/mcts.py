from __future__ import annotations
from typing import Optional
import math

import numpy as np
import ray


class Node:

    def __init__(
        self,
        state,
        prior: float,
        to_play: int = 0,
        action: tuple[int, int] = (-1, -1),
        parent: Optional[Node] = None
    ):
        self.state = state
        self.action = action
        self.to_play = to_play
        self.parent = parent
        self.reward_sum: float = 0.0
        self.N: int = 0  # Number of visits
        self.Q: float = 0.0  # Average reward
        self.P: float = prior  # Prior probability
        self.children: list[Node] = list()

    def __str__(self) -> str:
        return f"Node: {self.action} | N: {self.N} | Q: {self.Q} | P: {self.P} | To play: {self.to_play}"

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
        pb_c = math.log((1 + self.parent.N + C) / C) + 1
        return pb_c * self.P * np.sqrt(self.parent.N) / (1 + self.N)

    def PUCT(self, C: float) -> float:
        """The PUCT formula."""
        if self.parent is None:
            return 0
        return self._exploitation_value() + self._exploration_value(C)


class MCTS:

    def __init__(
        self,
        root: Node,
        model_actor,
        C: float,
        num_simulations: int,
        training: bool = False,
    ) -> None:
        self.root = root
        self.model_actor = model_actor
        self.C = C
        self.num_simulations = num_simulations
        self.training = training

    def search(self) -> tuple:
        for _ in range(self.num_simulations):
            node = self._select()
            if not node.state.is_terminal() and (node.is_leaf() and node.N > 0):
                self._expand(node)
                node = node.children[0]
            reward = self._playout(node)
            self._backpropagate(node, reward)

        policy = np.zeros((3, 3), dtype=np.float32)
        for child in self.root.children:
            policy[child.action] = child.N
        policy /= np.sum(policy)
        action = np.argmax(policy)

        return action, policy.flatten()

    def _select(self) -> Node:
        """Using the PUCT formula, select the best node to expand."""
        node = self.root
        while not node.is_leaf() and not node.state.is_terminal():
            node = max(node.children, key=lambda n: n.PUCT(self.C))
        return node

    def _expand(self, node: Node) -> None:
       """Expanding the node by adding children for all possible moves."""
       # Add Dirichlet noise for exploration
       if self.training:
           noise = np.random.dirichlet([0.03] * 9).reshape(3, 3)
       else:
           noise = np.zeros((3, 3))

       priors, _ = ray.get(self.model_actor.infer.remote(node.state.encode()))
       priors = np.reshape(priors, (3, 3))

       # Masking illegal actions and adding Dirichlet noise
       masked_priors = np.zeros((3, 3), dtype=np.float32)
       legal_actions = node.state.legal_actions()
       for action in legal_actions:
           masked_priors[action] = priors[action]
           if self.training:
               masked_priors[action] = (1 - 0.25) * masked_priors[action] + 0.25 * noise[action]

       # Normalize the priors
       masked_priors_sum = np.sum(masked_priors)
       if masked_priors_sum > 0:
           masked_priors /= masked_priors_sum

       # Create new nodes for all legal actions
       for action in legal_actions:
           new_state = node.state.step(action)
           prior = masked_priors[action].item()
           new_node = Node(new_state, prior, 1 - node.to_play, action, parent=node)
           node.children.append(new_node)

    def _playout(self, node: Node) -> float:
        """Basic playout function. Randomly selects moves until the game is finished."""
        if node.state.is_terminal():
            reward = node.state.reward(node.state.current_player)
        else:
            _, value = ray.get(self.model_actor.infer.remote(node.state.encode()))
            reward = value
        return reward

    def _backpropagate(self, node: Node, reward: float) -> None:
        """Backpropagating the reward up the tree."""
        while True:
            node.N += 1
            node.reward_sum += reward
            node.Q = node.reward_sum / node.N
            if node.parent is None:
                break
            node = node.parent
            reward = -reward
