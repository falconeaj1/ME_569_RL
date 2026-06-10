"""Ultimate Tic Tac Toe rules and state transitions.

The state implements the generic two-player protocol used by ``connect4.mcts``:
``current_player``, ``legal_actions()``, ``next_state()``, ``is_terminal``, and
``result_for()``. Actions are row-major integers from 0 to 80 on the 9x9 board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np


BOARD_SIZE = 9
LOCAL_SIZE = 3
PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0
LOCAL_ONGOING = 0
LOCAL_TIE = 2


class UltimateStatus(Enum):
    """Terminal/non-terminal status for the full Ultimate Tic Tac Toe game."""

    ONGOING = "ongoing"
    TIE = "tie"
    X_WIN = "x_win"
    O_WIN = "o_win"

    @property
    def is_terminal(self) -> bool:
        return self != UltimateStatus.ONGOING


@dataclass(frozen=True)
class UltimateMoveResult:
    """Details for an applied move."""

    action: int
    row: int
    col: int
    local_board: tuple[int, int]
    player: int
    status: UltimateStatus
    next_active_board: tuple[int, int] | None


def other_player(player: int) -> int:
    if int(player) not in {PLAYER_X, PLAYER_O}:
        raise ValueError("player must be 1 for X or -1 for O")
    return -int(player)


def player_label(player: int) -> str:
    if int(player) == PLAYER_X:
        return "X"
    if int(player) == PLAYER_O:
        return "O"
    raise ValueError("player must be 1 for X or -1 for O")


def status_for_winner(player: int) -> UltimateStatus:
    return UltimateStatus.X_WIN if int(player) == PLAYER_X else UltimateStatus.O_WIN


def winner_from_status(status: UltimateStatus) -> int | None:
    if status == UltimateStatus.X_WIN:
        return PLAYER_X
    if status == UltimateStatus.O_WIN:
        return PLAYER_O
    return None


def _empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def _empty_local_status() -> np.ndarray:
    return np.zeros((LOCAL_SIZE, LOCAL_SIZE), dtype=np.int8)


@dataclass(frozen=True)
class UltimateTicTacToeState:
    """Immutable Ultimate Tic Tac Toe state for search and UI code.

    ``local_status`` stores one value per 3x3 sub-board:
    ``0`` = still playable, ``1`` = X claimed it, ``-1`` = O claimed it,
    and ``2`` = tied/closed with no owner.
    """

    board: np.ndarray = field(default_factory=_empty_board)
    local_status: np.ndarray = field(default_factory=_empty_local_status)
    current_player: int = PLAYER_X
    active_board: tuple[int, int] | None = None
    status: UltimateStatus = UltimateStatus.ONGOING
    last_move: tuple[int, int] | None = None
    move_count: int = 0

    def __post_init__(self) -> None:
        if self.board.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError("ultimate tic tac toe board must be 9x9")
        if self.local_status.shape != (LOCAL_SIZE, LOCAL_SIZE):
            raise ValueError("local_status must be 3x3")
        if int(self.current_player) not in {PLAYER_X, PLAYER_O}:
            raise ValueError("current_player must be 1 for X or -1 for O")
        if self.active_board is not None:
            board_row, board_col = self.active_board
            if not (0 <= board_row < LOCAL_SIZE and 0 <= board_col < LOCAL_SIZE):
                raise ValueError("active_board must be a (row, col) pair in [0, 2]")

        object.__setattr__(self, "board", self.board.astype(np.int8, copy=False))
        object.__setattr__(self, "local_status", self.local_status.astype(np.int8, copy=False))
        object.__setattr__(self, "current_player", int(self.current_player))

    @classmethod
    def new(cls, starting_player: int = PLAYER_X) -> "UltimateTicTacToeState":
        return cls(current_player=starting_player)

    @classmethod
    def from_board(
        cls,
        board: Iterable[Iterable[int]] | np.ndarray,
        current_player: int = PLAYER_X,
        active_board: tuple[int, int] | None = None,
        last_move: tuple[int, int] | None = None,
    ) -> "UltimateTicTacToeState":
        array = np.array(board, dtype=np.int8)
        local_status = calculate_local_statuses(array)
        return cls.from_components(
            board=array,
            local_status=local_status,
            current_player=current_player,
            active_board=active_board,
            last_move=last_move,
        )

    @classmethod
    def from_components(
        cls,
        board: np.ndarray | None = None,
        local_status: np.ndarray | None = None,
        current_player: int = PLAYER_X,
        active_board: tuple[int, int] | None = None,
        status: UltimateStatus | None = None,
        last_move: tuple[int, int] | None = None,
        move_count: int | None = None,
    ) -> "UltimateTicTacToeState":
        board_array = _empty_board() if board is None else np.array(board, dtype=np.int8)
        local_array = (
            calculate_local_statuses(board_array)
            if local_status is None
            else np.array(local_status, dtype=np.int8)
        )
        resolved_status = calculate_status(local_array) if status is None else status
        resolved_move_count = int(np.count_nonzero(board_array)) if move_count is None else move_count
        if active_board is not None and int(local_array[active_board]) != LOCAL_ONGOING:
            active_board = None
        return cls(
            board=board_array,
            local_status=local_array,
            current_player=current_player,
            active_board=active_board,
            status=resolved_status,
            last_move=last_move,
            move_count=resolved_move_count,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def winner(self) -> int | None:
        return winner_from_status(self.status)

    @property
    def macro_claims(self) -> np.ndarray:
        """Return the 3x3 board of claimed sub-boards, with tied boards empty."""

        return np.where(np.abs(self.local_status) == 1, self.local_status, 0).astype(np.int8)

    @property
    def winning_boards(self) -> tuple[tuple[int, int], ...]:
        return tuple(find_winning_cells_3x3(self.macro_claims))

    def copy(self) -> "UltimateTicTacToeState":
        return UltimateTicTacToeState(
            board=self.board.copy(),
            local_status=self.local_status.copy(),
            current_player=self.current_player,
            active_board=self.active_board,
            status=self.status,
            last_move=self.last_move,
            move_count=self.move_count,
        )

    def legal_actions(self) -> list[int]:
        if self.is_terminal:
            return []

        candidate_boards: list[tuple[int, int]]
        if self.active_board is not None and int(self.local_status[self.active_board]) == LOCAL_ONGOING:
            candidate_boards = [self.active_board]
        else:
            candidate_boards = [
                (board_row, board_col)
                for board_row in range(LOCAL_SIZE)
                for board_col in range(LOCAL_SIZE)
                if int(self.local_status[board_row, board_col]) == LOCAL_ONGOING
            ]

        actions: list[int] = []
        for board_row, board_col in candidate_boards:
            row_start = board_row * LOCAL_SIZE
            col_start = board_col * LOCAL_SIZE
            for local_row in range(LOCAL_SIZE):
                for local_col in range(LOCAL_SIZE):
                    row = row_start + local_row
                    col = col_start + local_col
                    if int(self.board[row, col]) == EMPTY:
                        actions.append(row * BOARD_SIZE + col)
        return actions

    def is_legal_action(self, action: int) -> bool:
        return int(action) in self.legal_actions()

    def next_state(self, action: int) -> "UltimateTicTacToeState":
        action = int(action)
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"invalid ultimate tic tac toe action {action}")

        row, col = divmod(action, BOARD_SIZE)
        board_row, board_col = row // LOCAL_SIZE, col // LOCAL_SIZE
        player = self.current_player

        next_board = self.board.copy()
        next_local_status = self.local_status.copy()
        next_board[row, col] = player

        local_slice = local_board_view(next_board, board_row, board_col)
        next_local_status[board_row, board_col] = calculate_local_status(local_slice)
        next_status = calculate_status(next_local_status)
        target_board = (row % LOCAL_SIZE, col % LOCAL_SIZE)
        if next_status.is_terminal or int(next_local_status[target_board]) != LOCAL_ONGOING:
            next_active_board = None
        else:
            next_active_board = target_board

        next_player = other_player(player) if next_status == UltimateStatus.ONGOING else player
        return UltimateTicTacToeState(
            board=next_board,
            local_status=next_local_status,
            current_player=next_player,
            active_board=next_active_board,
            status=next_status,
            last_move=(row, col),
            move_count=self.move_count + 1,
        )

    def result_for(self, player: int) -> float:
        winner = self.winner
        if winner is None:
            return 0.0
        return 1.0 if winner == int(player) else -1.0

    def observation(self, perspective: int | None = None) -> np.ndarray:
        if perspective is None:
            return self.board.copy()
        return (self.board * int(perspective)).astype(np.int8, copy=True)

    def render_ascii(self) -> str:
        tokens = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        lines: list[str] = []
        for row in range(BOARD_SIZE):
            parts = []
            for board_col in range(LOCAL_SIZE):
                start = board_col * LOCAL_SIZE
                end = start + LOCAL_SIZE
                parts.append(" ".join(tokens[int(value)] for value in self.board[row, start:end]))
            lines.append(" | ".join(parts))
            if row in {2, 5}:
                lines.append("------+-------+------")
        return "\n".join(lines)


def action_to_row_col(action: int) -> tuple[int, int]:
    return divmod(int(action), BOARD_SIZE)


def row_col_to_action(row: int, col: int) -> int:
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError("row and col must be in [0, 8]")
    return int(row) * BOARD_SIZE + int(col)


def local_board_view(board: np.ndarray, board_row: int, board_col: int) -> np.ndarray:
    row_start = board_row * LOCAL_SIZE
    col_start = board_col * LOCAL_SIZE
    return board[row_start : row_start + LOCAL_SIZE, col_start : col_start + LOCAL_SIZE]


def calculate_local_statuses(board: np.ndarray) -> np.ndarray:
    array = np.array(board, dtype=np.int8)
    if array.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("ultimate tic tac toe board must be 9x9")
    statuses = _empty_local_status()
    for board_row in range(LOCAL_SIZE):
        for board_col in range(LOCAL_SIZE):
            statuses[board_row, board_col] = calculate_local_status(
                local_board_view(array, board_row, board_col)
            )
    return statuses


def calculate_local_status(local_board: np.ndarray) -> int:
    winner = find_winner_3x3(local_board)
    if winner is not None:
        return winner
    if np.all(local_board != EMPTY):
        return LOCAL_TIE
    return LOCAL_ONGOING


def calculate_status(local_status: np.ndarray) -> UltimateStatus:
    macro_claims = np.where(np.abs(local_status) == 1, local_status, 0).astype(np.int8)
    winner = find_winner_3x3(macro_claims)
    if winner is not None:
        return status_for_winner(winner)
    if np.all(local_status != LOCAL_ONGOING):
        return UltimateStatus.TIE
    return UltimateStatus.ONGOING


def find_winner_3x3(board: np.ndarray) -> int | None:
    cells = find_winning_cells_3x3(board)
    if not cells:
        return None
    row, col = cells[0]
    return int(board[row, col])


def find_winning_cells_3x3(board: np.ndarray) -> list[tuple[int, int]]:
    if board.shape != (LOCAL_SIZE, LOCAL_SIZE):
        raise ValueError("winner checks expect a 3x3 board")

    lines = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]

    for cells in lines:
        values = [int(board[row, col]) for row, col in cells]
        if values == [PLAYER_X, PLAYER_X, PLAYER_X]:
            return cells
        if values == [PLAYER_O, PLAYER_O, PLAYER_O]:
            return cells
    return []

