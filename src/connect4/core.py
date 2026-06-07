"""Core Connect Four game rules and state transitions.

This module is intentionally independent from pygame and Gymnasium. The same
state object powers interactive play, training environments, agents, and MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable

import numpy as np


DEFAULT_ROWS = 6
DEFAULT_COLS = 7
DEFAULT_CONNECT_N = 4


class Player(IntEnum):
    """Board values for the two players."""

    RED = 1
    YELLOW = -1

    @property
    def other(self) -> "Player":
        return Player.YELLOW if self == Player.RED else Player.RED

    @property
    def label(self) -> str:
        return "Red" if self == Player.RED else "Yellow"


class GameStatus(Enum):
    """Terminal/non-terminal game status."""

    ONGOING = "ongoing"
    TIE = "tie"
    RED_WIN = "red_win"
    YELLOW_WIN = "yellow_win"

    @property
    def is_terminal(self) -> bool:
        return self != GameStatus.ONGOING


@dataclass(frozen=True)
class MoveResult:
    """Details for a successfully applied move."""

    row: int
    col: int
    player: Player
    status: GameStatus


def status_for_winner(player: Player) -> GameStatus:
    return GameStatus.RED_WIN if player == Player.RED else GameStatus.YELLOW_WIN


def winner_from_status(status: GameStatus) -> Player | None:
    if status == GameStatus.RED_WIN:
        return Player.RED
    if status == GameStatus.YELLOW_WIN:
        return Player.YELLOW
    return None


def _empty_board(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.int8)


@dataclass
class ConnectFourState:
    """Mutable Connect Four state with copy helpers for agents/search.

    The board uses row 0 as the top row and row ``rows - 1`` as the bottom row.
    Empty cells are 0, red pieces are 1, and yellow pieces are -1.
    """

    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS
    connect_n: int = DEFAULT_CONNECT_N
    board: np.ndarray = field(default_factory=lambda: _empty_board(DEFAULT_ROWS, DEFAULT_COLS))
    current_player: Player = Player.RED
    status: GameStatus = GameStatus.ONGOING
    last_move: tuple[int, int] | None = None
    move_count: int = 0

    def __post_init__(self) -> None:
        expected_shape = (self.rows, self.cols)
        if self.board.shape != expected_shape:
            raise ValueError(f"board shape {self.board.shape} does not match {expected_shape}")
        if self.connect_n < 2:
            raise ValueError("connect_n must be at least 2")
        if self.connect_n > max(self.rows, self.cols):
            raise ValueError("connect_n must fit inside the board")
        if not isinstance(self.current_player, Player):
            self.current_player = Player(self.current_player)
        self.board = self.board.astype(np.int8, copy=False)

    @classmethod
    def new(
        cls,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        connect_n: int = DEFAULT_CONNECT_N,
        starting_player: Player = Player.RED,
    ) -> "ConnectFourState":
        return cls(
            rows=rows,
            cols=cols,
            connect_n=connect_n,
            board=_empty_board(rows, cols),
            current_player=starting_player,
        )

    @classmethod
    def from_board(
        cls,
        board: Iterable[Iterable[int]] | np.ndarray,
        current_player: Player = Player.RED,
        connect_n: int = DEFAULT_CONNECT_N,
        last_move: tuple[int, int] | None = None,
    ) -> "ConnectFourState":
        array = np.array(board, dtype=np.int8)
        if array.ndim != 2:
            raise ValueError("board must be a 2D array")
        rows, cols = array.shape
        status = calculate_status(array, connect_n)
        return cls(
            rows=rows,
            cols=cols,
            connect_n=connect_n,
            board=array,
            current_player=current_player,
            status=status,
            last_move=last_move,
            move_count=int(np.count_nonzero(array)),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def winner(self) -> Player | None:
        return winner_from_status(self.status)

    @property
    def winning_cells(self) -> tuple[tuple[int, int], ...]:
        """Cells that form the first detected winning streak."""

        cells = find_winning_cells(self.board, self.connect_n)
        return tuple(cells)

    def copy(self) -> "ConnectFourState":
        return ConnectFourState(
            rows=self.rows,
            cols=self.cols,
            connect_n=self.connect_n,
            board=self.board.copy(),
            current_player=self.current_player,
            status=self.status,
            last_move=self.last_move,
            move_count=self.move_count,
        )

    def legal_actions(self) -> list[int]:
        if self.is_terminal:
            return []
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.cols, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def is_legal_action(self, action: int) -> bool:
        return int(action) in self.legal_actions()

    def drop_piece(self, action: int) -> MoveResult:
        """Apply a move for ``current_player`` and advance the game."""

        col = int(action)
        if self.is_terminal:
            raise ValueError("cannot move after the game is over")
        if col < 0 or col >= self.cols:
            raise ValueError(f"column must be in [0, {self.cols - 1}], got {col}")
        if self.board[0, col] != 0:
            raise ValueError(f"column {col} is full")

        row = self._drop_row(col)
        player = self.current_player
        self.board[row, col] = int(player)
        self.last_move = (row, col)
        self.move_count += 1
        self.status = calculate_status(self.board, self.connect_n)

        if self.status == GameStatus.ONGOING:
            self.current_player = player.other

        return MoveResult(row=row, col=col, player=player, status=self.status)

    def next_state(self, action: int) -> "ConnectFourState":
        state = self.copy()
        state.drop_piece(action)
        return state

    def result_for(self, player: int | Player) -> float:
        """Return terminal value from ``player`` perspective.

        Win = 1, loss = -1, tie or non-terminal = 0. MCTS calls this after
        rollout termination, while Gym envs use it to report sparse rewards.
        """

        winner = self.winner
        if winner is None:
            return 0.0
        return 1.0 if winner == Player(player) else -1.0

    def observation(self, perspective: Player | int | None = None) -> np.ndarray:
        """Return a board copy, optionally normalized to one player's perspective."""

        if perspective is None:
            return self.board.copy()
        return (self.board * int(Player(perspective))).astype(np.int8, copy=True)

    def render_ascii(self) -> str:
        """Render the board as plain text for debugging and tests."""

        tokens = {int(Player.RED): "R", int(Player.YELLOW): "Y", 0: "."}
        lines = [" ".join(tokens[int(cell)] for cell in row) for row in self.board]
        lines.append(" ".join(str(col) for col in range(self.cols)))
        return "\n".join(lines)

    def _drop_row(self, col: int) -> int:
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError(f"column {col} is full")


def calculate_status(board: np.ndarray, connect_n: int = DEFAULT_CONNECT_N) -> GameStatus:
    winner = find_winner(board, connect_n)
    if winner is not None:
        return status_for_winner(winner)
    if np.all(board != 0):
        return GameStatus.TIE
    return GameStatus.ONGOING


def find_winner(board: np.ndarray, connect_n: int = DEFAULT_CONNECT_N) -> Player | None:
    cells = find_winning_cells(board, connect_n)
    if not cells:
        return None
    row, col = cells[0]
    return Player(int(board[row, col]))


def find_winning_cells(board: np.ndarray, connect_n: int = DEFAULT_CONNECT_N) -> list[tuple[int, int]]:
    rows, cols = board.shape
    directions = ((0, 1), (1, 0), (1, 1), (-1, 1))

    for row in range(rows):
        for col in range(cols):
            value = int(board[row, col])
            if value == 0:
                continue
            for row_delta, col_delta in directions:
                end_row = row + (connect_n - 1) * row_delta
                end_col = col + (connect_n - 1) * col_delta
                if not (0 <= end_row < rows and 0 <= end_col < cols):
                    continue
                cells = [
                    (row + i * row_delta, col + i * col_delta)
                    for i in range(connect_n)
                ]
                if all(int(board[cell_row, cell_col]) == value for cell_row, cell_col in cells):
                    return cells
    return []
