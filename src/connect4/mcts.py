"""Monte Carlo Tree Search for deterministic two-player games."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class TwoPlayerState(Protocol):
    """State protocol required by :class:`MCTS`."""

    current_player: int

    @property
    def is_terminal(self) -> bool:
        ...

    def legal_actions(self) -> list[int]:
        ...

    def next_state(self, action: int) -> "TwoPlayerState":
        ...

    def result_for(self, player: int) -> float:
        ...


RolloutPolicy = Callable[[TwoPlayerState, random.Random], int]
ValueEvaluator = Callable[[TwoPlayerState, int], float]


def random_rollout_policy(state: TwoPlayerState, rng: random.Random) -> int:
    """Sample uniformly from legal actions during rollout."""

    return rng.choice(state.legal_actions())


@dataclass
class MCTSNode:
    """One node in the search tree."""

    state: TwoPlayerState
    parent: "MCTSNode | None" = None
    action: int | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_actions and not self.state.is_terminal:
            self.untried_actions = list(self.state.legal_actions())

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass(frozen=True)
class MCTSResult:
    """Search output for the root position."""

    action: int
    action_stats: dict[int, dict[str, float]]
    root_visits: int


class MCTSSearch:
    """Incremental MCTS search rooted at one position."""

    def __init__(
        self,
        root_state: TwoPlayerState,
        exploration_weight: float = math.sqrt(2),
        seed: int | None = None,
        max_rollout_steps: int = 1_000,
        rollout_policy: RolloutPolicy = random_rollout_policy,
        value_evaluator: ValueEvaluator | None = None,
        value_depth: int | None = None,
    ) -> None:
        legal = root_state.legal_actions()
        if not legal:
            raise ValueError("MCTS requires at least one legal action")
        if value_depth is not None and value_depth < 0:
            raise ValueError("value_depth must be non-negative")
        self.legal_actions = legal
        self.root_player = int(root_state.current_player)
        self.root = MCTSNode(state=root_state)
        self.exploration_weight = exploration_weight
        self.max_rollout_steps = max_rollout_steps
        self.rollout_policy = rollout_policy
        self.value_evaluator = value_evaluator
        self.value_depth = value_depth
        self.rng = random.Random(seed)

    @property
    def visits(self) -> int:
        return self.root.visits

    def run_iterations(self, iterations: int) -> None:
        if iterations < 1:
            return
        for _ in range(iterations):
            self.run_iteration()

    def run_for_seconds(self, seconds: float, min_iterations: int = 0, max_iterations: int | None = None) -> None:
        if seconds <= 0 and min_iterations <= 0:
            return

        deadline = time.perf_counter() + max(0.0, seconds)
        completed = 0
        while True:
            if max_iterations is not None and completed >= max_iterations:
                break
            if completed >= min_iterations and time.perf_counter() >= deadline:
                break
            self.run_iteration()
            completed += 1

    def run_iteration(self) -> None:
        node = self._select_or_expand(self.root)
        value = self._rollout_value(node.state)
        self._backpropagate(node, value)

    def result(self) -> MCTSResult:
        children = self.root.children
        if not children:
            action = self.rng.choice(self.legal_actions)
            return MCTSResult(action=action, action_stats={}, root_visits=self.root.visits)

        best_child = max(
            children.values(),
            key=lambda child: (child.visits, child.mean_value, -int(child.action)),
        )
        return MCTSResult(
            action=int(best_child.action),
            action_stats=self._action_stats(children),
            root_visits=self.root.visits,
        )

    def _select_or_expand(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal:
            if node.untried_actions:
                return self._expand(node)
            node = self._select_child(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child = MCTSNode(
            state=node.state.next_state(action),
            parent=node,
            action=action,
        )
        node.children[action] = child
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_parent = math.log(max(1, node.visits))
        maximizing_for_root = int(node.state.current_player) == self.root_player

        def uct_score(child: MCTSNode) -> float:
            exploitation = child.mean_value if maximizing_for_root else -child.mean_value
            exploration = self.exploration_weight * math.sqrt(log_parent / child.visits)
            return exploitation + exploration

        return max(node.children.values(), key=uct_score)

    def _rollout_value(self, state: TwoPlayerState) -> float:
        current = state
        steps = 0
        while not current.is_terminal:
            if self.value_evaluator is not None and self.value_depth is not None and steps >= self.value_depth:
                return self.value_evaluator(current, self.root_player)
            legal = current.legal_actions()
            if not legal:
                break
            action = self.rollout_policy(current, self.rng)
            if action not in legal:
                action = self.rng.choice(legal)
            current = current.next_state(action)
            steps += 1
            if steps >= self.max_rollout_steps:
                break
        if current.is_terminal:
            return current.result_for(self.root_player)
        if self.value_evaluator is not None:
            return self.value_evaluator(current, self.root_player)
        return current.result_for(self.root_player)

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    @staticmethod
    def _action_stats(children: dict[int, MCTSNode]) -> dict[int, dict[str, float]]:
        return {
            action: {"visits": child.visits, "mean_value": child.mean_value}
            for action, child in sorted(children.items())
        }


class MCTS:
    """Adversarial UCT search.

    Values are stored from the root player's perspective. At root-player nodes
    selection prefers high values; at opponent nodes it prefers low root values.
    """

    def __init__(
        self,
        iterations: int = 1_000,
        exploration_weight: float = math.sqrt(2),
        seed: int | None = None,
        max_rollout_steps: int = 1_000,
        time_limit_seconds: float | None = None,
        rollout_policy: RolloutPolicy = random_rollout_policy,
        value_evaluator: ValueEvaluator | None = None,
        value_depth: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if value_depth is not None and value_depth < 0:
            raise ValueError("value_depth must be non-negative")
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.max_rollout_steps = max_rollout_steps
        self.time_limit_seconds = time_limit_seconds
        self.rollout_policy = rollout_policy
        self.value_evaluator = value_evaluator
        self.value_depth = value_depth
        self.rng = random.Random(seed)

    def search(self, root_state: TwoPlayerState) -> MCTSResult:
        search = MCTSSearch(
            root_state=root_state,
            exploration_weight=self.exploration_weight,
            seed=self.rng.randrange(2**31),
            max_rollout_steps=self.max_rollout_steps,
            rollout_policy=self.rollout_policy,
            value_evaluator=self.value_evaluator,
            value_depth=self.value_depth,
        )
        min_iterations = len(search.legal_actions)
        if self.time_limit_seconds is not None:
            search.run_for_seconds(
                self.time_limit_seconds,
                min_iterations=min_iterations,
                max_iterations=self.iterations,
            )
        else:
            search.run_iterations(self.iterations)
        return search.result()

    def _select_or_expand(self, node: MCTSNode, root_player: int) -> MCTSNode:
        while not node.state.is_terminal:
            if node.untried_actions:
                return self._expand(node)
            node = self._select_child(node, root_player)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child = MCTSNode(
            state=node.state.next_state(action),
            parent=node,
            action=action,
        )
        node.children[action] = child
        return child

    def _select_child(self, node: MCTSNode, root_player: int) -> MCTSNode:
        log_parent = math.log(max(1, node.visits))
        maximizing_for_root = int(node.state.current_player) == root_player

        def uct_score(child: MCTSNode) -> float:
            exploitation = child.mean_value if maximizing_for_root else -child.mean_value
            exploration = self.exploration_weight * math.sqrt(log_parent / child.visits)
            return exploitation + exploration

        return max(node.children.values(), key=uct_score)

    def _rollout(self, state: TwoPlayerState) -> TwoPlayerState:
        current = state
        steps = 0
        while not current.is_terminal:
            legal = current.legal_actions()
            if not legal:
                break
            current = current.next_state(self.rng.choice(legal))
            steps += 1
            if steps >= self.max_rollout_steps:
                break
        return current

    @staticmethod
    def _backpropagate(node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    @staticmethod
    def _action_stats(children: dict[int, MCTSNode]) -> dict[int, dict[str, float]]:
        return {
            action: {"visits": child.visits, "mean_value": child.mean_value}
            for action, child in sorted(children.items())
        }
