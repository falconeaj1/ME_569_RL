"""Computer players and helper evaluation functions for Connect Four."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from connect4.core import ConnectFourState, Player
from connect4.mcts import MCTS, RolloutPolicy, random_rollout_policy
from connect4.value_model import ValueModelEvaluator, ValueNetwork


class Agent(Protocol):
    """Minimal interface shared by all computer-controlled players."""

    name: str

    def select_action(self, state: ConnectFourState) -> int:
        """Choose a legal action for the given state."""
        ...


@dataclass
class RandomAgent:
    """Agent that samples uniformly from legal columns."""

    seed: int | None = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, state: ConnectFourState) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("no legal actions available")
        return self.rng.choice(legal)


@dataclass
class FirstLegalAgent:
    """Deterministic baseline that always chooses the leftmost legal column."""

    name: str = "first-legal"

    def select_action(self, state: ConnectFourState) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("no legal actions available")
        return legal[0]


@dataclass
class HeuristicAgent:
    """One-ply tactical agent with a simple board evaluator."""

    seed: int | None = None
    name: str = "heuristic"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, state: ConnectFourState) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("no legal actions available")

        player = state.current_player

        winning_move = immediate_winning_action(state, player)
        if winning_move is not None:
            return winning_move

        blocking_move = immediate_winning_action(state, player.other)
        if blocking_move is not None:
            return blocking_move

        scored_actions = [
            (score_action(state, action, player), action)
            for action in legal
        ]
        best_score = max(score for score, _ in scored_actions)
        best_actions = [action for score, action in scored_actions if score == best_score]
        return self.rng.choice(best_actions)


@dataclass
class BadHeuristicAgent:
    """A deliberately weak agent that prefers low-scoring legal moves."""

    seed: int | None = None
    name: str = "bad-heuristic"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, state: ConnectFourState) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("no legal actions available")

        player = state.current_player
        scored_actions = [
            (score_action(state, action, player), action)
            for action in legal
        ]
        worst_score = min(score for score, _ in scored_actions)
        worst_actions = [action for score, action in scored_actions if score == worst_score]
        return self.rng.choice(worst_actions)


@dataclass
class MCTSAgent:
    """Agent that runs Monte Carlo Tree Search at each turn."""

    iterations: int = 800
    exploration_weight: float = 2 ** 0.5
    max_rollout_steps: int = 1_000
    time_limit_seconds: float | None = None
    rollout_policy: str = "random"
    value_model_path: str | None = None
    value_depth: int | None = None
    seed: int | None = None
    name: str = "mcts"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.rollout_action = rollout_policy_from_name(self.rollout_policy)
        self.value_evaluator = None
        if self.value_model_path is not None:
            self.value_evaluator = ValueModelEvaluator(ValueNetwork.load(Path(self.value_model_path)))

    def select_action(self, state: ConnectFourState) -> int:
        search = MCTS(
            iterations=self.iterations,
            exploration_weight=self.exploration_weight,
            seed=self.rng.randrange(2**31),
            max_rollout_steps=self.max_rollout_steps,
            time_limit_seconds=self.time_limit_seconds,
            rollout_policy=self.rollout_action,
            value_evaluator=self.value_evaluator,
            value_depth=self.value_depth,
        )
        return search.search(state.copy()).action


def rollout_policy_from_name(name: str) -> RolloutPolicy:
    """Return a rollout action selector by name."""

    normalized = name.strip().lower().replace("_", "-")
    if normalized == "random":
        return random_rollout_policy
    if normalized in {"heuristic", "heuristic-guided"}:
        return heuristic_rollout_policy
    if normalized in {"value", "value-cutoff"}:
        return random_rollout_policy
    raise ValueError(f"unknown rollout policy '{name}'")


def heuristic_rollout_policy(state: ConnectFourState, rng: random.Random) -> int:
    """Choose rollout moves with the same tactical rules as ``HeuristicAgent``."""

    legal = state.legal_actions()
    if not legal:
        raise ValueError("no legal actions available")

    player = state.current_player
    winning_move = immediate_winning_action(state, player)
    if winning_move is not None:
        return winning_move

    blocking_move = immediate_winning_action(state, player.other)
    if blocking_move is not None:
        return blocking_move

    scored_actions = [(score_action(state, action, player), action) for action in legal]
    best_score = max(score for score, _ in scored_actions)
    best_actions = [action for score, action in scored_actions if score == best_score]
    return rng.choice(best_actions)


def immediate_winning_action(state: ConnectFourState, player: Player) -> int | None:
    """Return a legal action that wins immediately for ``player``, if one exists."""

    for action in state.legal_actions():
        next_state = state.copy()
        next_state.current_player = player
        next_state.drop_piece(action)
        if next_state.winner == player:
            return action
    return None


def score_action(state: ConnectFourState, action: int, player: Player) -> int:
    """Score the board that would result from ``player`` taking ``action``."""

    next_state = state.copy()
    next_state.drop_piece(action)
    return evaluate_board(next_state.board, player, connect_n=state.connect_n)


def evaluate_board(board: np.ndarray, player: Player, connect_n: int = 4) -> int:
    """Estimate board strength from ``player`` perspective.

    The score rewards center control and open two/three-in-a-row windows while
    penalizing comparable opponent threats. It is a tactical baseline, not a
    learned value function.
    """

    player_value = int(player)
    opponent_value = int(player.other)
    score = 0

    center_col = board.shape[1] // 2
    score += int(np.count_nonzero(board[:, center_col] == player_value)) * 3

    for window in iter_windows(board, connect_n):
        own = window.count(player_value)
        opponent = window.count(opponent_value)
        empty = window.count(0)

        if own == connect_n:
            score += 100_000
        elif own == connect_n - 1 and empty == 1:
            score += 80
        elif own == connect_n - 2 and empty == 2:
            score += 15

        if opponent == connect_n:
            score -= 100_000
        elif opponent == connect_n - 1 and empty == 1:
            score -= 100
        elif opponent == connect_n - 2 and empty == 2:
            score -= 12

    return score


def iter_windows(board: np.ndarray, connect_n: int) -> list[list[int]]:
    """Return every horizontal, vertical, and diagonal window of length ``connect_n``."""

    rows, cols = board.shape
    windows: list[list[int]] = []

    for row in range(rows):
        for col in range(cols - connect_n + 1):
            windows.append([int(board[row, col + i]) for i in range(connect_n)])

    for row in range(rows - connect_n + 1):
        for col in range(cols):
            windows.append([int(board[row + i, col]) for i in range(connect_n)])

    for row in range(rows - connect_n + 1):
        for col in range(cols - connect_n + 1):
            windows.append([int(board[row + i, col + i]) for i in range(connect_n)])

    for row in range(connect_n - 1, rows):
        for col in range(cols - connect_n + 1):
            windows.append([int(board[row - i, col + i]) for i in range(connect_n)])

    return windows


def agent_from_name(
    name: str,
    seed: int | None = None,
    mcts_iterations: int = 800,
    mcts_exploration_weight: float = 2 ** 0.5,
    mcts_max_rollout_steps: int = 1_000,
    mcts_time_limit_seconds: float | None = None,
    mcts_rollout_policy: str = "random",
    mcts_value_model_path: str | None = None,
    mcts_value_depth: int | None = None,
) -> Agent:
    """Build an agent from a CLI/config-friendly name.

    This factory intentionally handles selection-time MCTS settings only. Model
    training settings should live in a separate training config once learned
    agents are added.
    """

    normalized = name.strip().lower().replace("_", "-")
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized in {"bad", "bad-heuristic"}:
        return BadHeuristicAgent(seed=seed)
    if normalized in {"heuristic", "good"}:
        return HeuristicAgent(seed=seed)
    if normalized == "mcts":
        return MCTSAgent(
            iterations=mcts_iterations,
            exploration_weight=mcts_exploration_weight,
            max_rollout_steps=mcts_max_rollout_steps,
            time_limit_seconds=mcts_time_limit_seconds,
            rollout_policy=mcts_rollout_policy,
            value_model_path=mcts_value_model_path,
            value_depth=mcts_value_depth,
            seed=seed,
        )
    if normalized in {"first", "first-legal"}:
        return FirstLegalAgent()
    raise ValueError(f"unknown agent '{name}'")
