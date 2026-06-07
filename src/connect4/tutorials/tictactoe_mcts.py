"""Small Tic Tac Toe state used to inspect and debug MCTS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from connect4.mcts import MCTS


PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0


@dataclass(frozen=True)
class TicTacToeState:
    """Immutable Tic Tac Toe state implementing the MCTS game protocol."""

    board: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.int8))
    current_player: int = PLAYER_X

    def __post_init__(self) -> None:
        if self.board.shape != (3, 3):
            raise ValueError("tic tac toe board must be 3x3")
        object.__setattr__(self, "board", self.board.astype(np.int8, copy=False))

    @classmethod
    def from_rows(cls, rows: list[list[int]], current_player: int = PLAYER_X) -> "TicTacToeState":
        return cls(board=np.array(rows, dtype=np.int8), current_player=current_player)

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or not self.legal_actions()

    @property
    def winner(self) -> int | None:
        lines = []
        lines.extend(self.board[row, :] for row in range(3))
        lines.extend(self.board[:, col] for col in range(3))
        lines.append(np.array([self.board[i, i] for i in range(3)]))
        lines.append(np.array([self.board[i, 2 - i] for i in range(3)]))

        for line in lines:
            total = int(np.sum(line))
            if total == 3:
                return PLAYER_X
            if total == -3:
                return PLAYER_O
        return None

    def legal_actions(self) -> list[int]:
        if self.winner is not None:
            return []
        return [idx for idx, value in enumerate(self.board.reshape(-1)) if int(value) == EMPTY]

    def next_state(self, action: int) -> "TicTacToeState":
        if action not in self.legal_actions():
            raise ValueError(f"invalid tic tac toe action {action}")
        row, col = divmod(int(action), 3)
        next_board = self.board.copy()
        next_board[row, col] = self.current_player
        return TicTacToeState(board=next_board, current_player=-self.current_player)

    def result_for(self, player: int) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == int(player) else -1.0

    def render_ascii(self) -> str:
        tokens = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        rows = [" ".join(tokens[int(cell)] for cell in row) for row in self.board]
        return "\n".join(rows)


def run_self_play(iterations: int, seed: int | None, verbose: bool = False) -> TicTacToeState:
    state = TicTacToeState()
    move_number = 1

    while not state.is_terminal:
        result = MCTS(iterations=iterations, seed=seed).search(state)
        player = "X" if state.current_player == PLAYER_X else "O"
        state = state.next_state(result.action)

        if verbose:
            print(f"\nMove {move_number}: {player} chooses action {result.action}")
            print_action_stats(result.action_stats, verbose)
            print(state.render_ascii())
        move_number += 1

    if state.winner == PLAYER_X and verbose:
        print("\nX wins")
    elif state.winner == PLAYER_O and verbose:
        print("\nO wins")
    elif verbose:
        print("\nTie")
    return state


def print_action_stats(action_stats: dict[int, dict[str, float]], verbose: bool = False) -> None:
    if not action_stats:
        return
    summary = []
    for action, stats in action_stats.items():
        summary.append(
            f"{action}: visits={int(stats['visits'])}, value={stats['mean_value']:.2f}"
        )
    if verbose:
        print("  " + " | ".join(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny MCTS tutorial on Tic Tac Toe.")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Will print per game results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_self_play(iterations=args.iterations, seed=args.seed, verbose=args.verbose)


if __name__ == "__main__":
    main()
