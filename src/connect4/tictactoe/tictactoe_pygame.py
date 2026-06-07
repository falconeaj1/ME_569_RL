"""Pygame visualizer for Tic Tac Toe agents and state values."""

from __future__ import annotations

import argparse
import functools
import random
import time
from dataclasses import dataclass
from typing import Callable

import pygame
import pygame.gfxdraw

from connect4.mcts import MCTS
from connect4.tictactoe.tictactoe_mcts import EMPTY, PLAYER_O, PLAYER_X, TicTacToeState


BOARD_PIXELS = 540
PANEL_WIDTH = 340
HEADER_HEIGHT = 72
WINDOW_WIDTH = BOARD_PIXELS + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_PIXELS + HEADER_HEIGHT
CELL_SIZE = BOARD_PIXELS // 3

BACKGROUND = (22, 24, 29)
BOARD_BG = (238, 240, 243)
PANEL_BG = (32, 35, 42)
GRID = (33, 38, 46)
X_COLOR = (225, 71, 67)
O_COLOR = (64, 107, 214)
WHITE = (245, 245, 245)
MUTED = (166, 171, 182)
BUTTON_BG = (56, 61, 73)
BUTTON_ACTIVE = (79, 118, 183)
GOOD = (70, 190, 120)
BAD = (220, 85, 80)
NEUTRAL = (210, 214, 221)
ITERATION_CHOICES = [20, 50, 100, 200, 500, 1_000, 5_000, 100_000]
TIME_CHOICES: list[float | None] = [None, 0.1, 0.25, 0.5, 1.0, 2.0]


@dataclass
class Button:
    label: str | Callable[[], str]
    rect: pygame.Rect
    action: Callable[[], None]
    active: Callable[[], bool] = lambda: False


def draw_smooth_circle(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int,
    width: int = 8,
) -> None:
    for offset in range(width):
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius - offset, color)


def winner_label(winner: int | None) -> str:
    if winner == PLAYER_X:
        return "X"
    if winner == PLAYER_O:
        return "O"
    return "Tie"


@functools.cache
def minimax_value(board_key: tuple[int, ...], current_player: int, perspective: int) -> float:
    """Return perfect-play value from ``perspective`` for a Tic Tac Toe state."""

    board = [board_key[i : i + 3] for i in range(0, 9, 3)]
    state = TicTacToeState.from_rows([list(row) for row in board], current_player=current_player)
    if state.is_terminal:
        return state.result_for(perspective)

    child_values = [
        minimax_value(
            tuple(int(v) for v in state.next_state(action).board.reshape(-1)),
            -current_player,
            perspective,
        )
        for action in state.legal_actions()
    ]
    if current_player == perspective:
        return max(child_values)
    return min(child_values)


def state_key(state: TicTacToeState) -> tuple[int, ...]:
    return tuple(int(v) for v in state.board.reshape(-1))


class TicTacToePygameApp:
    """Interactive Tic Tac Toe visualizer with exact state evaluation."""

    def __init__(
        self,
        mode: str,
        ai: str,
        human_player: int,
        mcts_iterations: int,
        mcts_time_limit: float | None,
        seed: int | None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Tic Tac Toe MCTS")

        self.mode = mode
        self.ai = ai
        self.human_player = human_player
        self.mcts_iterations = mcts_iterations
        self.mcts_time_limit = mcts_time_limit
        self.rng = random.Random(seed)
        self.seed = seed
        self.eval_enabled = True
        self.state = TicTacToeState()
        self.last_ai_move_at = 0.0
        self.last_move_text = ""
        self.last_search_visits: int | None = None

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 26, bold=True)
        self.medium_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 15)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
        self.buttons: list[Button] = []
        self._layout_buttons()

    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._advance_ai_if_needed()
            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key in {pygame.K_ESCAPE, pygame.K_q}:
            return False
        if key == pygame.K_r:
            self._reset()
        elif key == pygame.K_e:
            self.eval_enabled = not self.eval_enabled
        elif key in {pygame.K_LEFTBRACKET, pygame.K_MINUS}:
            self._step_iterations(-1)
        elif key in {pygame.K_RIGHTBRACKET, pygame.K_EQUALS}:
            self._step_iterations(1)
        elif key == pygame.K_COMMA:
            self._step_time_limit(-1)
        elif key == pygame.K_PERIOD:
            self._step_time_limit(1)
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                button.action()
                return

        if self.state.is_terminal or self._current_actor_is_ai():
            return
        if pos[0] >= BOARD_PIXELS or pos[1] < HEADER_HEIGHT:
            return

        col = pos[0] // CELL_SIZE
        row = (pos[1] - HEADER_HEIGHT) // CELL_SIZE
        action = row * 3 + col
        if action in self.state.legal_actions():
            self.state = self.state.next_state(action)
            self.last_ai_move_at = time.monotonic()

    def _advance_ai_if_needed(self) -> None:
        if self.state.is_terminal or not self._current_actor_is_ai():
            return
        if time.monotonic() - self.last_ai_move_at < 0.25:
            return

        start = time.perf_counter()
        action, visits = self._select_ai_action()
        elapsed = time.perf_counter() - start
        self.state = self.state.next_state(action)
        actor = "X" if -self.state.current_player == PLAYER_X else "O"
        self.last_search_visits = visits
        suffix = "" if visits is None else f", {visits} iters"
        self.last_move_text = f"{actor} {self.ai} played {action} in {elapsed:.2f}s{suffix}"
        self.last_ai_move_at = time.monotonic()

    def _select_ai_action(self) -> tuple[int, int | None]:
        legal = self.state.legal_actions()
        if self.ai == "random":
            return self.rng.choice(legal), None

        result = MCTS(
            iterations=self.mcts_iterations,
            seed=self.rng.randrange(1_000_000_000),
            time_limit_seconds=self.mcts_time_limit,
        ).search(self.state)
        return result.action, result.root_visits

    def _current_actor_is_ai(self) -> bool:
        if self.mode == "human-human":
            return False
        if self.mode == "ai-ai":
            return True
        return self.state.current_player != self.human_player

    def _reset(self) -> None:
        self.state = TicTacToeState()
        self.last_ai_move_at = 0.0
        self.last_move_text = ""
        self.last_search_visits = None

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._reset()

    def _set_ai(self, ai: str) -> None:
        self.ai = ai
        self._reset()

    def _set_human_player(self, player: int) -> None:
        self.human_player = player
        self._reset()

    def _toggle_eval(self) -> None:
        self.eval_enabled = not self.eval_enabled

    def _set_mcts_iterations(self, iterations: int, time_limit: float | None) -> None:
        self.mcts_iterations = iterations
        self.mcts_time_limit = time_limit
        self._reset()

    def _step_iterations(self, direction: int) -> None:
        try:
            index = ITERATION_CHOICES.index(self.mcts_iterations)
        except ValueError:
            closest = min(range(len(ITERATION_CHOICES)), key=lambda i: abs(ITERATION_CHOICES[i] - self.mcts_iterations))
            index = closest
        next_index = max(0, min(len(ITERATION_CHOICES) - 1, index + direction))
        self.mcts_iterations = ITERATION_CHOICES[next_index]
        self._reset()

    def _step_time_limit(self, direction: int) -> None:
        try:
            index = TIME_CHOICES.index(self.mcts_time_limit)
        except ValueError:
            numeric_time = 0.0 if self.mcts_time_limit is None else self.mcts_time_limit
            index = min(
                range(len(TIME_CHOICES)),
                key=lambda i: abs((0.0 if TIME_CHOICES[i] is None else float(TIME_CHOICES[i])) - numeric_time),
            )
        next_index = max(0, min(len(TIME_CHOICES) - 1, index + direction))
        self.mcts_time_limit = TIME_CHOICES[next_index]
        self._reset()

    def _layout_buttons(self) -> None:
        x = BOARD_PIXELS + 18
        y = 176
        full_width = PANEL_WIDTH - 36
        half_width = (full_width - 8) // 2
        height = 25
        row_gap = 5
        section_gap = 8

        def add_full(label: str | Callable[[], str], action: Callable[[], None], active: Callable[[], bool] = lambda: False) -> None:
            nonlocal y
            self.buttons.append(Button(label, pygame.Rect(x, y, full_width, height), action, active))
            y += height + row_gap

        def add_pair(
            left: tuple[str | Callable[[], str], Callable[[], None], Callable[[], bool]],
            right: tuple[str | Callable[[], str], Callable[[], None], Callable[[], bool]],
        ) -> None:
            nonlocal y
            self.buttons.append(Button(left[0], pygame.Rect(x, y, half_width, height), left[1], left[2]))
            self.buttons.append(Button(right[0], pygame.Rect(x + half_width + 8, y, half_width, height), right[1], right[2]))
            y += height + row_gap

        add_full("Reset", self._reset)
        y += section_gap
        add_pair(
            ("Human-Human", lambda: self._set_mode("human-human"), lambda: self.mode == "human-human"),
            ("Human-AI", lambda: self._set_mode("human-ai"), lambda: self.mode == "human-ai"),
        )
        add_full("AI-AI", lambda: self._set_mode("ai-ai"), lambda: self.mode == "ai-ai")
        y += section_gap
        add_pair(
            ("Human X", lambda: self._set_human_player(PLAYER_X), lambda: self.human_player == PLAYER_X),
            ("Human O", lambda: self._set_human_player(PLAYER_O), lambda: self.human_player == PLAYER_O),
        )
        y += section_gap
        add_pair(
            ("AI random", lambda: self._set_ai("random"), lambda: self.ai == "random"),
            ("AI MCTS", lambda: self._set_ai("mcts"), lambda: self.ai == "mcts"),
        )
        y += section_gap
        add_pair(
            (lambda: f"Iters - ({self.mcts_iterations})", lambda: self._step_iterations(-1), lambda: False),
            (lambda: f"Iters + ({self.mcts_iterations})", lambda: self._step_iterations(1), lambda: False),
        )
        add_pair(
            (lambda: f"Time - ({self._time_label()})", lambda: self._step_time_limit(-1), lambda: False),
            (lambda: f"Time + ({self._time_label()})", lambda: self._step_time_limit(1), lambda: False),
        )
        add_full("Timed 1s strong", lambda: self._set_mcts_iterations(100_000, 1.0), lambda: self.mcts_time_limit == 1.0)
        y += section_gap
        add_full("Evaluator overlay (E)", self._toggle_eval, lambda: self.eval_enabled)

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_header()
        self._draw_board()
        self._draw_panel()

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, BACKGROUND, pygame.Rect(0, 0, BOARD_PIXELS, HEADER_HEIGHT))
        self.screen.blit(self.font.render(self._status_text(), True, WHITE), (18, 10))
        self.screen.blit(self.small_font.render("Click square. R reset. E evaluator. Q quit.", True, MUTED), (18, 43))

    def _draw_board(self) -> None:
        board_rect = pygame.Rect(0, HEADER_HEIGHT, BOARD_PIXELS, BOARD_PIXELS)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect)

        if self.eval_enabled and not self.state.is_terminal:
            self._draw_evaluation_shading()

        for i in range(1, 3):
            x = i * CELL_SIZE
            y = HEADER_HEIGHT + i * CELL_SIZE
            pygame.draw.line(self.screen, GRID, (x, HEADER_HEIGHT + 18), (x, HEADER_HEIGHT + BOARD_PIXELS - 18), 7)
            pygame.draw.line(self.screen, GRID, (18, y), (BOARD_PIXELS - 18, y), 7)

        for row in range(3):
            for col in range(3):
                value = int(self.state.board[row, col])
                if value == PLAYER_X:
                    self._draw_x(row, col)
                elif value == PLAYER_O:
                    self._draw_o(row, col)

    def _draw_evaluation_shading(self) -> None:
        player = self.state.current_player
        for action in self.state.legal_actions():
            next_state = self.state.next_state(action)
            value = minimax_value(state_key(next_state), next_state.current_player, player)
            row, col = divmod(action, 3)
            rect = pygame.Rect(col * CELL_SIZE + 10, HEADER_HEIGHT + row * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
            if value > 0:
                color = GOOD
            elif value < 0:
                color = BAD
            else:
                color = NEUTRAL
            overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
            overlay.fill((*color, 80))
            self.screen.blit(overlay, rect)
            label = self.medium_font.render(f"{value:+.0f}", True, GRID)
            self.screen.blit(label, (rect.x + 10, rect.y + 8))

    def _draw_x(self, row: int, col: int) -> None:
        margin = 48
        x0 = col * CELL_SIZE + margin
        y0 = HEADER_HEIGHT + row * CELL_SIZE + margin
        x1 = (col + 1) * CELL_SIZE - margin
        y1 = HEADER_HEIGHT + (row + 1) * CELL_SIZE - margin
        pygame.draw.line(self.screen, X_COLOR, (x0, y0), (x1, y1), 12)
        pygame.draw.line(self.screen, X_COLOR, (x1, y0), (x0, y1), 12)
        pygame.draw.aaline(self.screen, X_COLOR, (x0, y0), (x1, y1))
        pygame.draw.aaline(self.screen, X_COLOR, (x1, y0), (x0, y1))

    def _draw_o(self, row: int, col: int) -> None:
        center = (col * CELL_SIZE + CELL_SIZE // 2, HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2)
        draw_smooth_circle(self.screen, O_COLOR, center, 52, width=10)

    def _draw_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, pygame.Rect(BOARD_PIXELS, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        self.screen.blit(self.font.render("Tic Tac Toe", True, WHITE), (BOARD_PIXELS + 18, 16))

        y = 55
        x_value = minimax_value(state_key(self.state), self.state.current_player, PLAYER_X)
        details = [
            f"Mode: {self.mode}",
            f"AI: {self.ai}",
            f"Budget: {self.mcts_iterations} iters / {self._time_label()}",
            "Stops at first cap hit",
            f"Last search: {self.last_search_visits or '-'} iters",
            f"Perfect value for X: {x_value:+.0f}",
        ]
        for detail in details:
            self.screen.blit(self.small_font.render(detail, True, MUTED), (BOARD_PIXELS + 18, y))
            y += 18

        for button in self.buttons:
            color = BUTTON_ACTIVE if button.active() else BUTTON_BG
            pygame.draw.rect(self.screen, color, button.rect, border_radius=5)
            label_text = button.label() if callable(button.label) else button.label
            label = self.small_font.render(label_text, True, WHITE)
            self.screen.blit(label, (button.rect.x + 9, button.rect.y + 4))

        if self.last_move_text:
            self.screen.blit(self.tiny_font.render(self.last_move_text[:42], True, WHITE), (BOARD_PIXELS + 18, WINDOW_HEIGHT - 24))

    def _status_text(self) -> str:
        if self.state.is_terminal:
            winner = self.state.winner
            if winner is None:
                return "Tie game"
            return f"{winner_label(winner)} wins"

        player = "X" if self.state.current_player == PLAYER_X else "O"
        actor = "AI" if self._current_actor_is_ai() else "Human"
        return f"{player} to move: {actor}"

    def _time_label(self) -> str:
        return "off" if self.mcts_time_limit is None else f"{self.mcts_time_limit:g}s"


def parse_player(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"x", "1"}:
        return PLAYER_X
    if normalized in {"o", "-1"}:
        return PLAYER_O
    raise argparse.ArgumentTypeError("player must be x or o")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play and inspect Tic Tac Toe agents.")
    parser.add_argument("--mode", choices=["human-human", "human-ai", "ai-ai"], default="human-ai")
    parser.add_argument("--ai", choices=["random", "mcts"], default="mcts")
    parser.add_argument("--human-player", type=parse_player, default=PLAYER_X)
    parser.add_argument("--mcts-iterations", type=int, default=200)
    parser.add_argument("--mcts-time-limit", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TicTacToePygameApp(
        mode=args.mode,
        ai=args.ai,
        human_player=args.human_player,
        mcts_iterations=args.mcts_iterations,
        mcts_time_limit=args.mcts_time_limit,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
