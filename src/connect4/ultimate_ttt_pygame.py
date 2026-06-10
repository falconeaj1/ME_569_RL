"""Interactive pygame app for Ultimate Tic Tac Toe with MCTS search."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Callable

import pygame
import pygame.gfxdraw

from connect4.mcts import MCTS, MCTSSearch
from connect4.ultimate_ttt import (
    BOARD_SIZE,
    LOCAL_ONGOING,
    LOCAL_SIZE,
    LOCAL_TIE,
    PLAYER_O,
    PLAYER_X,
    UltimateStatus,
    UltimateTicTacToeState,
    action_to_row_col,
    player_label,
    row_col_to_action,
)


CELL_SIZE = 68
BOARD_PIXELS = BOARD_SIZE * CELL_SIZE
HEADER_HEIGHT = 72
EVAL_BAR_HEIGHT = 98
PANEL_WIDTH = 430
WINDOW_WIDTH = BOARD_PIXELS + PANEL_WIDTH
WINDOW_HEIGHT = HEADER_HEIGHT + EVAL_BAR_HEIGHT + BOARD_PIXELS
BOARD_TOP = HEADER_HEIGHT + EVAL_BAR_HEIGHT

BACKGROUND = (22, 24, 29)
BOARD_BG = (239, 241, 244)
PANEL_BG = (32, 35, 42)
CELL_LINE = (82, 88, 98)
LOCAL_LINE = (25, 29, 36)
ACTIVE_BOARD = (52, 142, 235)
LEGAL_DOT = (74, 151, 105)
LAST_MOVE = (255, 218, 95)
X_COLOR = (222, 62, 58)
O_COLOR = (57, 100, 214)
TIE_COLOR = (140, 147, 158)
WHITE = (246, 247, 249)
MUTED = (169, 174, 184)
BUTTON_BG = (56, 61, 73)
BUTTON_ACTIVE = (79, 118, 183)
BUTTON_DISABLED = (43, 47, 56)
BUTTON_DISABLED_TEXT = (120, 125, 136)
GOOD = (54, 194, 122)
BAD = (235, 112, 74)
NEUTRAL = (78, 84, 96)
EVAL_BG = (18, 21, 27)

MCTS_PRESETS = {
    "fast": {"iterations": 250, "time_limit": 0.25},
    "medium": {"iterations": 1_000, "time_limit": 1.0},
    "strong": {"iterations": 4_000, "time_limit": 3.0},
}
ITERATION_CHOICES = [100, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
TIME_CHOICES: list[float | None] = [None, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


@dataclass
class Button:
    label: str | Callable[[], str]
    rect: pygame.Rect
    action: Callable[[], None]
    active: Callable[[], bool] = lambda: False
    enabled: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class ActionEvaluation:
    action: int
    mcts_visits: int = 0
    mcts_value: float = 0.0
    is_mcts_choice: bool = False


class UltimateTicTacToePygameApp:
    """Playable Ultimate Tic Tac Toe app backed by the shared MCTS engine."""

    def __init__(
        self,
        mode: str,
        ai: str,
        human_player: int,
        seed: int | None,
        mcts_iterations: int,
        mcts_exploration_weight: float,
        mcts_max_rollout_steps: int,
        mcts_time_limit_seconds: float | None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Ultimate Tic Tac Toe")

        self.mode = mode
        self.ai = ai
        self.human_player = human_player
        self.seed = seed
        self.rng = random.Random(seed)
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_weight = mcts_exploration_weight
        self.mcts_max_rollout_steps = mcts_max_rollout_steps
        self.mcts_time_limit_seconds = mcts_time_limit_seconds
        self.mcts_profile = self._profile_from_settings(mcts_iterations, mcts_time_limit_seconds)
        self.eval_enabled = True
        self.eval_time_slice_seconds = 0.02
        self.eval_search_key: tuple[bytes, bytes, int, tuple[int, int] | None] | None = None
        self.eval_search: MCTSSearch | None = None
        self.history: list[UltimateTicTacToeState] = []
        self.state = UltimateTicTacToeState.new()
        self.last_agent_move_at = 0.0
        self.last_agent_description = ""

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 26, bold=True)
        self.medium_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 15)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
        self.claim_font = pygame.font.SysFont("Arial", 118, bold=True)
        self.buttons: list[Button] = []
        self.panel_details_y = 0
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

            self._advance_agent_if_needed()
            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key in {pygame.K_ESCAPE, pygame.K_q}:
            return False
        if key == pygame.K_r:
            self._reset()
        elif key == pygame.K_u:
            self._undo()
        elif key == pygame.K_e:
            self._toggle_eval()
        elif key == pygame.K_LEFTBRACKET:
            self._step_iterations(-1)
        elif key == pygame.K_RIGHTBRACKET:
            self._step_iterations(1)
        elif key == pygame.K_MINUS:
            self._step_time_limit(-1)
        elif key == pygame.K_EQUALS:
            self._step_time_limit(1)
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                if button.enabled():
                    button.action()
                return

        if self.state.status != UltimateStatus.ONGOING or self._current_actor_is_ai():
            return
        if pos[0] >= BOARD_PIXELS or pos[1] < BOARD_TOP:
            return

        col = pos[0] // CELL_SIZE
        row = (pos[1] - BOARD_TOP) // CELL_SIZE
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        action = row_col_to_action(row, col)
        if self.state.is_legal_action(action):
            self._record_history()
            self.state = self.state.next_state(action)
            self.last_agent_move_at = time.monotonic()
            self.last_agent_description = ""
            self._clear_eval_cache()

    def _advance_agent_if_needed(self) -> None:
        if self.state.status != UltimateStatus.ONGOING or not self._current_actor_is_ai():
            return
        if time.monotonic() - self.last_agent_move_at < 0.25:
            return

        start = time.perf_counter()
        action, visits = self._select_agent_action()
        elapsed = time.perf_counter() - start
        if self.state.is_legal_action(action):
            actor = player_label(self.state.current_player)
            self._record_history()
            self.state = self.state.next_state(action)
            suffix = "" if visits is None else f", {visits} visits"
            self.last_agent_description = f"{actor} {self.ai} played {format_action(action)} in {elapsed:.2f}s{suffix}"
        self.last_agent_move_at = time.monotonic()
        self._clear_eval_cache()

    def _select_agent_action(self) -> tuple[int, int | None]:
        legal = self.state.legal_actions()
        if not legal:
            raise ValueError("no legal actions available")
        if self.ai == "random":
            return self.rng.choice(legal), None

        result = MCTS(
            iterations=self.mcts_iterations,
            exploration_weight=self.mcts_exploration_weight,
            seed=self.rng.randrange(2**31),
            max_rollout_steps=self.mcts_max_rollout_steps,
            time_limit_seconds=self.mcts_time_limit_seconds,
        ).search(self.state.copy())
        return result.action, result.root_visits

    def _current_actor_is_ai(self) -> bool:
        if self.mode == "human-human":
            return False
        if self.mode == "ai-ai":
            return True
        return self.state.current_player != self.human_player

    def _reset(self) -> None:
        self.state = UltimateTicTacToeState.new()
        self.history = []
        self.last_agent_move_at = 0.0
        self.last_agent_description = ""
        self._clear_eval_cache()

    def _record_history(self) -> None:
        self.history.append(self.state.copy())

    def _can_undo(self) -> bool:
        return self.mode != "ai-ai" and bool(self.history)

    def _undo(self) -> None:
        if not self._can_undo():
            return

        self.state = self.history.pop()
        if self.mode == "human-ai":
            while self.history and self._current_actor_is_ai():
                self.state = self.history.pop()

        self.last_agent_move_at = time.monotonic()
        self.last_agent_description = ""
        self._clear_eval_cache()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._reset()

    def _set_ai(self, name: str) -> None:
        self.ai = name
        self._reset()

    def _set_human_player(self, player: int) -> None:
        self.human_player = player
        self._reset()

    def _set_mcts_preset(self, preset: str) -> None:
        settings = MCTS_PRESETS[preset]
        self.mcts_profile = preset
        self.mcts_iterations = int(settings["iterations"])
        self.mcts_time_limit_seconds = float(settings["time_limit"])
        self._reset()

    def _set_mcts_custom(self) -> None:
        self.mcts_profile = "custom"

    def _step_iterations(self, direction: int) -> None:
        if not self._custom_mcts_controls_enabled():
            return
        current = self.mcts_iterations
        if current in ITERATION_CHOICES:
            index = ITERATION_CHOICES.index(current)
        else:
            index = min(range(len(ITERATION_CHOICES)), key=lambda i: abs(ITERATION_CHOICES[i] - current))
        next_index = max(0, min(len(ITERATION_CHOICES) - 1, index + direction))
        self.mcts_iterations = ITERATION_CHOICES[next_index]
        self._clear_eval_cache()

    def _step_time_limit(self, direction: int) -> None:
        if not self._custom_mcts_controls_enabled():
            return
        current = self.mcts_time_limit_seconds

        def distance(value: float | None) -> float:
            if current is None:
                return 0.0 if value is None else float("inf")
            if value is None:
                return float("inf")
            return abs(value - current)

        if current in TIME_CHOICES:
            index = TIME_CHOICES.index(current)
        else:
            index = min(range(len(TIME_CHOICES)), key=lambda i: distance(TIME_CHOICES[i]))
        next_index = max(0, min(len(TIME_CHOICES) - 1, index + direction))
        self.mcts_time_limit_seconds = TIME_CHOICES[next_index]
        self._clear_eval_cache()

    def _profile_from_settings(self, iterations: int, time_limit: float | None) -> str:
        for profile, settings in MCTS_PRESETS.items():
            if iterations == int(settings["iterations"]) and time_limit == float(settings["time_limit"]):
                return profile
        return "custom"

    def _ai_controls_enabled(self) -> bool:
        return self.mode != "human-human"

    def _mcts_controls_enabled(self) -> bool:
        return self._ai_controls_enabled() and self.ai == "mcts"

    def _custom_mcts_controls_enabled(self) -> bool:
        return self._mcts_controls_enabled() and self.mcts_profile == "custom"

    def _toggle_eval(self) -> None:
        self.eval_enabled = not self.eval_enabled
        self._clear_eval_cache()

    def _clear_eval_cache(self) -> None:
        self.eval_search_key = None
        self.eval_search = None

    def _layout_buttons(self) -> None:
        x = BOARD_PIXELS + 18
        y = 66
        full_width = PANEL_WIDTH - 36
        half_width = (full_width - 8) // 2
        height = 24
        row_gap = 5
        section_gap = 8

        def add_full(
            label: str | Callable[[], str],
            action: Callable[[], None],
            active: Callable[[], bool] = lambda: False,
            enabled: Callable[[], bool] = lambda: True,
        ) -> None:
            nonlocal y
            self.buttons.append(Button(label, pygame.Rect(x, y, full_width, height), action, active, enabled))
            y += height + row_gap

        def add_pair(
            left: tuple[str | Callable[[], str], Callable[[], None], Callable[[], bool], Callable[[], bool]],
            right: tuple[str | Callable[[], str], Callable[[], None], Callable[[], bool], Callable[[], bool]] | None = None,
        ) -> None:
            nonlocal y
            self.buttons.append(Button(left[0], pygame.Rect(x, y, half_width, height), left[1], left[2], left[3]))
            if right is not None:
                self.buttons.append(
                    Button(
                        right[0],
                        pygame.Rect(x + half_width + 8, y, half_width, height),
                        right[1],
                        right[2],
                        right[3],
                    )
                )
            y += height + row_gap

        add_pair(
            ("Reset", self._reset, lambda: False, lambda: True),
            ("Undo", self._undo, lambda: False, self._can_undo),
        )
        y += section_gap
        add_pair(
            ("Human-Human", lambda: self._set_mode("human-human"), lambda: self.mode == "human-human", lambda: True),
            ("Human-AI", lambda: self._set_mode("human-ai"), lambda: self.mode == "human-ai", lambda: True),
        )
        add_full("AI-AI", lambda: self._set_mode("ai-ai"), lambda: self.mode == "ai-ai")
        y += section_gap
        add_pair(
            ("Human X", lambda: self._set_human_player(PLAYER_X), lambda: self.human_player == PLAYER_X, lambda: self.mode == "human-ai"),
            ("Human O", lambda: self._set_human_player(PLAYER_O), lambda: self.human_player == PLAYER_O, lambda: self.mode == "human-ai"),
        )
        y += section_gap
        add_pair(
            ("AI random", lambda: self._set_ai("random"), lambda: self._ai_controls_enabled() and self.ai == "random", self._ai_controls_enabled),
            ("AI MCTS", lambda: self._set_ai("mcts"), lambda: self._ai_controls_enabled() and self.ai == "mcts", self._ai_controls_enabled),
        )
        y += section_gap
        add_pair(
            ("Fast 250/0.25s", lambda: self._set_mcts_preset("fast"), lambda: self._preset_is_active("fast"), self._mcts_controls_enabled),
            ("Medium 1000/1s", lambda: self._set_mcts_preset("medium"), lambda: self._preset_is_active("medium"), self._mcts_controls_enabled),
        )
        add_pair(
            ("Strong 4000/3s", lambda: self._set_mcts_preset("strong"), lambda: self._preset_is_active("strong"), self._mcts_controls_enabled),
            ("Custom", self._set_mcts_custom, lambda: self._mcts_controls_enabled() and self.mcts_profile == "custom", self._mcts_controls_enabled),
        )
        add_pair(
            (lambda: f"Iters - ({self.mcts_iterations})", lambda: self._step_iterations(-1), lambda: False, self._custom_mcts_controls_enabled),
            (lambda: f"Iters + ({self.mcts_iterations})", lambda: self._step_iterations(1), lambda: False, self._custom_mcts_controls_enabled),
        )
        add_pair(
            (lambda: f"Time - ({self._time_label()})", lambda: self._step_time_limit(-1), lambda: False, self._custom_mcts_controls_enabled),
            (lambda: f"Time + ({self._time_label()})", lambda: self._step_time_limit(1), lambda: False, self._custom_mcts_controls_enabled),
        )
        y += section_gap
        add_full("Evaluation", self._toggle_eval, lambda: self.eval_enabled)
        self.panel_details_y = y + 6

    def _preset_is_active(self, preset: str) -> bool:
        return self._mcts_controls_enabled() and self.mcts_profile == preset

    def _time_label(self) -> str:
        return "off" if self.mcts_time_limit_seconds is None else f"{self.mcts_time_limit_seconds:g}s"

    def _evaluations(self) -> dict[int, ActionEvaluation]:
        if not self.eval_enabled or self.state.status != UltimateStatus.ONGOING:
            return {}

        key = (
            self.state.board.tobytes(),
            self.state.local_status.tobytes(),
            int(self.state.current_player),
            self.state.active_board,
        )
        if key != self.eval_search_key:
            self.eval_search = MCTSSearch(
                root_state=self.state.copy(),
                exploration_weight=self.mcts_exploration_weight,
                seed=self.seed,
                max_rollout_steps=self.mcts_max_rollout_steps,
            )
            self.eval_search_key = key

        if self.eval_search is None:
            return {}

        min_iterations = len(self.state.legal_actions()) if self.eval_search.visits == 0 else 0
        self.eval_search.run_for_seconds(
            self.eval_time_slice_seconds,
            min_iterations=min_iterations,
        )
        result = self.eval_search.result()
        evaluations: dict[int, ActionEvaluation] = {}
        for action, stats in result.action_stats.items():
            evaluations[action] = ActionEvaluation(
                action=action,
                mcts_visits=int(stats["visits"]),
                mcts_value=float(stats["mean_value"]),
                is_mcts_choice=action == result.action,
            )
        return evaluations

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        evaluations = self._evaluations()
        self._draw_header()
        self._draw_evaluation_bar(evaluations)
        self._draw_board(evaluations)
        self._draw_panel(evaluations)

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, BACKGROUND, pygame.Rect(0, 0, BOARD_PIXELS, HEADER_HEIGHT))
        self.screen.blit(self.font.render(self._status_text(), True, WHITE), (18, 12))
        active = "any open board" if self.state.active_board is None else f"local board {self.state.active_board[0] + 1},{self.state.active_board[1] + 1}"
        self.screen.blit(self.small_font.render(f"Target: {active}", True, MUTED), (18, 48))

    def _draw_evaluation_bar(self, evaluations: dict[int, ActionEvaluation]) -> None:
        rect = pygame.Rect(0, HEADER_HEIGHT, BOARD_PIXELS, EVAL_BAR_HEIGHT)
        pygame.draw.rect(self.screen, EVAL_BG, rect)
        pygame.draw.line(self.screen, (39, 43, 52), rect.bottomleft, rect.bottomright, 2)

        if not self.eval_enabled:
            self.screen.blit(self.small_font.render("Evaluation off", True, MUTED), (18, rect.y + 30))
            return
        if self.state.status != UltimateStatus.ONGOING:
            self.screen.blit(self.small_font.render("Game over", True, MUTED), (18, rect.y + 30))
            return
        if self.eval_search is None or self.eval_search.visits <= 0:
            self.screen.blit(self.small_font.render("Evaluation starting", True, MUTED), (18, rect.y + 30))
            return

        score = self._x_perspective_eval_score()
        gauge = pygame.Rect(78, rect.y + 26, BOARD_PIXELS - 156, 20)
        midpoint = gauge.centerx
        pygame.draw.rect(self.screen, O_COLOR, pygame.Rect(gauge.x, gauge.y, midpoint - gauge.x, gauge.height), border_radius=5)
        pygame.draw.rect(self.screen, X_COLOR, pygame.Rect(midpoint, gauge.y, gauge.right - midpoint, gauge.height), border_radius=5)
        pygame.draw.rect(self.screen, WHITE, gauge, width=1, border_radius=5)
        pygame.draw.line(self.screen, WHITE, (midpoint, gauge.y - 5), (midpoint, gauge.bottom + 5), 1)

        if score is None:
            marker_x = midpoint
            score_label = "even"
        else:
            marker_x = midpoint + int(max(-1.0, min(1.0, score)) * gauge.width / 2)
            if score > 0.05:
                score_label = f"X {score:+.2f}"
            elif score < -0.05:
                score_label = f"O {-score:+.2f}"
            else:
                score_label = "even"
        pygame.draw.circle(self.screen, WHITE, (marker_x, gauge.centery), 9)
        pygame.draw.circle(self.screen, NEUTRAL, (marker_x, gauge.centery), 5)

        self.screen.blit(self.small_font.render("O", True, O_COLOR), (gauge.x - 28, gauge.y + 1))
        self.screen.blit(self.small_font.render("X", True, X_COLOR), (gauge.right + 14, gauge.y + 1))
        self.screen.blit(self.tiny_font.render(f"Visits: {self.eval_search.visits}", True, MUTED), (gauge.x, gauge.bottom + 12))
        self.screen.blit(self.tiny_font.render(f"Position: {score_label}", True, MUTED), (gauge.x + 116, gauge.bottom + 12))

        leader = self._evaluation_leader(evaluations)
        if leader is not None:
            self.screen.blit(
                self.tiny_font.render(
                    f"Top move: {format_action(leader.action)}  value {leader.mcts_value:+.2f}  N {leader.mcts_visits}",
                    True,
                    MUTED,
                ),
                (gauge.x + 250, gauge.bottom + 12),
            )

    def _draw_board(self, evaluations: dict[int, ActionEvaluation]) -> None:
        board_rect = pygame.Rect(0, BOARD_TOP, BOARD_PIXELS, BOARD_PIXELS)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect)

        self._draw_local_backgrounds()
        self._draw_legal_cells()
        self._draw_pieces()
        self._draw_action_evaluations(evaluations)
        self._draw_local_claims()
        self._draw_last_move()
        self._draw_grid()
        self._draw_macro_winning_line()

    def _draw_local_backgrounds(self) -> None:
        for board_row in range(LOCAL_SIZE):
            for board_col in range(LOCAL_SIZE):
                rect = local_board_rect(board_row, board_col)
                status = int(self.state.local_status[board_row, board_col])
                if self.state.active_board == (board_row, board_col):
                    draw_alpha_rect(self.screen, ACTIVE_BOARD, rect, 42)
                    pygame.draw.rect(self.screen, ACTIVE_BOARD, rect.inflate(-8, -8), width=3, border_radius=5)
                elif status == LOCAL_ONGOING and self.state.active_board is None:
                    draw_alpha_rect(self.screen, LEGAL_DOT, rect, 12)

                if status == PLAYER_X:
                    draw_alpha_rect(self.screen, X_COLOR, rect, 34)
                elif status == PLAYER_O:
                    draw_alpha_rect(self.screen, O_COLOR, rect, 34)
                elif status == LOCAL_TIE:
                    draw_alpha_rect(self.screen, TIE_COLOR, rect, 26)

    def _draw_legal_cells(self) -> None:
        if self.state.status != UltimateStatus.ONGOING:
            return
        legal_actions = set(self.state.legal_actions())
        for action in legal_actions:
            row, col = action_to_row_col(action)
            center = cell_center(row, col)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 5, LEGAL_DOT)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 5, LEGAL_DOT)

    def _draw_pieces(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                value = int(self.state.board[row, col])
                center = cell_center(row, col)
                if value == PLAYER_X:
                    draw_x(self.screen, center, CELL_SIZE // 2 - 12, X_COLOR, 6)
                elif value == PLAYER_O:
                    draw_o(self.screen, center, CELL_SIZE // 2 - 12, O_COLOR, 6)

    def _draw_action_evaluations(self, evaluations: dict[int, ActionEvaluation]) -> None:
        if not evaluations:
            return
        max_visits = max((evaluation.mcts_visits for evaluation in evaluations.values()), default=0)
        if max_visits <= 0:
            return

        for evaluation in evaluations.values():
            row, col = action_to_row_col(evaluation.action)
            rect = pygame.Rect(col * CELL_SIZE + 9, BOARD_TOP + row * CELL_SIZE + CELL_SIZE - 14, CELL_SIZE - 18, 5)
            share = evaluation.mcts_visits / max_visits
            fill_width = max(2, int(rect.width * share))
            pygame.draw.rect(self.screen, (216, 220, 226), rect, border_radius=2)
            pygame.draw.rect(self.screen, value_color(evaluation.mcts_value), pygame.Rect(rect.x, rect.y, fill_width, rect.height), border_radius=2)
            if evaluation.is_mcts_choice:
                center = cell_center(row, col)
                pygame.draw.circle(self.screen, WHITE, center, CELL_SIZE // 2 - 7, width=3)
                pygame.draw.circle(self.screen, GOOD, center, CELL_SIZE // 2 - 12, width=2)

    def _draw_local_claims(self) -> None:
        for board_row in range(LOCAL_SIZE):
            for board_col in range(LOCAL_SIZE):
                status = int(self.state.local_status[board_row, board_col])
                rect = local_board_rect(board_row, board_col)
                if status == PLAYER_X:
                    self._draw_claim_text("X", X_COLOR, rect)
                elif status == PLAYER_O:
                    self._draw_claim_text("O", O_COLOR, rect)
                elif status == LOCAL_TIE:
                    self._draw_claim_text("T", TIE_COLOR, rect)

    def _draw_claim_text(self, text: str, color: tuple[int, int, int], rect: pygame.Rect) -> None:
        label = self.claim_font.render(text, True, color)
        label.set_alpha(72)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_last_move(self) -> None:
        if self.state.last_move is None:
            return
        row, col = self.state.last_move
        center = cell_center(row, col)
        pygame.draw.circle(self.screen, LAST_MOVE, center, CELL_SIZE // 2 - 5, width=3)
        pygame.draw.circle(self.screen, LAST_MOVE, center, 5)

    def _draw_grid(self) -> None:
        for index in range(BOARD_SIZE + 1):
            width = 5 if index % LOCAL_SIZE == 0 else 1
            color = LOCAL_LINE if index % LOCAL_SIZE == 0 else CELL_LINE
            pos = index * CELL_SIZE
            pygame.draw.line(self.screen, color, (pos, BOARD_TOP), (pos, BOARD_TOP + BOARD_PIXELS), width)
            pygame.draw.line(self.screen, color, (0, BOARD_TOP + pos), (BOARD_PIXELS, BOARD_TOP + pos), width)

    def _draw_macro_winning_line(self) -> None:
        if self.state.winner is None:
            return
        cells = self.state.winning_boards
        if not cells:
            return
        centers = [
            (
                board_col * LOCAL_SIZE * CELL_SIZE + (LOCAL_SIZE * CELL_SIZE) // 2,
                BOARD_TOP + board_row * LOCAL_SIZE * CELL_SIZE + (LOCAL_SIZE * CELL_SIZE) // 2,
            )
            for board_row, board_col in cells
        ]
        pygame.draw.lines(self.screen, WHITE, False, centers, 16)
        pygame.draw.lines(self.screen, GOOD, False, centers, 7)

    def _draw_panel(self, evaluations: dict[int, ActionEvaluation]) -> None:
        panel_rect = pygame.Rect(BOARD_PIXELS, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        self.screen.blit(self.font.render("Ultimate TTT", True, WHITE), (BOARD_PIXELS + 18, 16))

        for button in self.buttons:
            enabled = button.enabled()
            color = BUTTON_DISABLED if not enabled else BUTTON_ACTIVE if button.active() else BUTTON_BG
            pygame.draw.rect(self.screen, color, button.rect, border_radius=5)
            text_color = WHITE if enabled else BUTTON_DISABLED_TEXT
            label = button.label() if callable(button.label) else button.label
            self.screen.blit(self.small_font.render(label, True, text_color), (button.rect.x + 9, button.rect.y + 4))

        y = self.panel_details_y
        details = [
            f"Mode: {self.mode}",
            f"AI: {self.ai if self._ai_controls_enabled() else 'off'}",
            f"Human: {player_label(self.human_player) if self.mode == 'human-ai' else 'n/a'}",
            f"MCTS: {self.mcts_iterations} / {self._time_label()}",
            f"Profile: {self.mcts_profile if self._mcts_controls_enabled() else 'off'}",
            f"Evaluation: {'on' if self.eval_enabled else 'off'}",
        ]
        for detail in details:
            self.screen.blit(self.small_font.render(detail, True, MUTED), (BOARD_PIXELS + 18, y))
            y += 19

        y += 10
        y = self._draw_macro_map(y)
        y += 12

        leader = self._evaluation_leader(evaluations)
        if leader is not None:
            lines = [
                "Search leader",
                f"Move {format_action(leader.action)}",
                f"Value {leader.mcts_value:+.2f}",
                f"Visits {leader.mcts_visits}",
            ]
            for index, line in enumerate(lines):
                font = self.medium_font if index == 0 else self.small_font
                color = WHITE if index == 0 else MUTED
                self.screen.blit(font.render(line, True, color), (BOARD_PIXELS + 18, y))
                y += 21

        if self.last_agent_description:
            clipped = self.last_agent_description[:52]
            self.screen.blit(self.tiny_font.render(clipped, True, WHITE), (BOARD_PIXELS + 18, WINDOW_HEIGHT - 22))

    def _draw_macro_map(self, y: int) -> int:
        self.screen.blit(self.medium_font.render("Local boards", True, WHITE), (BOARD_PIXELS + 18, y))
        y += 28
        size = 48
        gap = 7
        x0 = BOARD_PIXELS + 18
        for board_row in range(LOCAL_SIZE):
            for board_col in range(LOCAL_SIZE):
                rect = pygame.Rect(x0 + board_col * (size + gap), y + board_row * (size + gap), size, size)
                status = int(self.state.local_status[board_row, board_col])
                color = (45, 49, 58)
                if status == PLAYER_X:
                    color = X_COLOR
                elif status == PLAYER_O:
                    color = O_COLOR
                elif status == LOCAL_TIE:
                    color = TIE_COLOR
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                if self.state.active_board == (board_row, board_col):
                    pygame.draw.rect(self.screen, WHITE, rect, width=3, border_radius=5)
                text = local_status_label(status)
                label = self.medium_font.render(text, True, WHITE)
                self.screen.blit(label, label.get_rect(center=rect.center))
        return y + LOCAL_SIZE * size + (LOCAL_SIZE - 1) * gap

    def _x_perspective_eval_score(self) -> float | None:
        if self.eval_search is None or self.eval_search.visits <= 0:
            return None
        player_to_move_value = self.eval_search.root.mean_value
        x_score = player_to_move_value * int(self.state.current_player)
        return max(-1.0, min(1.0, x_score))

    def _evaluation_leader(self, evaluations: dict[int, ActionEvaluation]) -> ActionEvaluation | None:
        if not evaluations:
            return None
        return max(evaluations.values(), key=lambda evaluation: (evaluation.is_mcts_choice, evaluation.mcts_visits))

    def _status_text(self) -> str:
        if self.state.status == UltimateStatus.TIE:
            return "Tie game"
        if self.state.winner is not None:
            return f"{player_label(self.state.winner)} wins"

        actor = "AI" if self._current_actor_is_ai() else "Human"
        return f"{player_label(self.state.current_player)} to move: {actor}"


def draw_alpha_rect(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    alpha: int,
) -> None:
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill((*color, alpha))
    surface.blit(overlay, rect.topleft)


def draw_x(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    x, y = center
    pygame.draw.line(surface, color, (x - radius, y - radius), (x + radius, y + radius), width)
    pygame.draw.line(surface, color, (x + radius, y - radius), (x - radius, y + radius), width)


def draw_o(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    for offset in range(width):
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius - offset, color)


def cell_center(row: int, col: int) -> tuple[int, int]:
    return (
        col * CELL_SIZE + CELL_SIZE // 2,
        BOARD_TOP + row * CELL_SIZE + CELL_SIZE // 2,
    )


def local_board_rect(board_row: int, board_col: int) -> pygame.Rect:
    return pygame.Rect(
        board_col * LOCAL_SIZE * CELL_SIZE,
        BOARD_TOP + board_row * LOCAL_SIZE * CELL_SIZE,
        LOCAL_SIZE * CELL_SIZE,
        LOCAL_SIZE * CELL_SIZE,
    )


def local_status_label(status: int) -> str:
    if status == PLAYER_X:
        return "X"
    if status == PLAYER_O:
        return "O"
    if status == LOCAL_TIE:
        return "T"
    return "."


def value_color(value: float) -> tuple[int, int, int]:
    score = max(-1.0, min(1.0, value))
    if abs(score) < 0.05:
        return NEUTRAL
    source = GOOD if score > 0 else BAD
    strength = 0.35 + 0.65 * abs(score)
    return tuple(
        int(NEUTRAL[channel] * (1.0 - strength) + source[channel] * strength)
        for channel in range(3)
    )


def format_action(action: int) -> str:
    row, col = action_to_row_col(action)
    return f"r{row + 1}c{col + 1}"


def parse_player(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"x", "1", "first"}:
        return PLAYER_X
    if normalized in {"o", "-1", "second"}:
        return PLAYER_O
    raise argparse.ArgumentTypeError("player must be X or O")


def run_game(
    mode: str = "human-ai",
    ai: str = "mcts",
    human_player: int = PLAYER_X,
    seed: int | None = None,
    mcts_iterations: int = 1_000,
    mcts_exploration_weight: float = 2 ** 0.5,
    mcts_max_rollout_steps: int = 90,
    mcts_time_limit_seconds: float | None = 1.0,
) -> None:
    app = UltimateTicTacToePygameApp(
        mode=mode,
        ai=ai,
        human_player=human_player,
        seed=seed,
        mcts_iterations=mcts_iterations,
        mcts_exploration_weight=mcts_exploration_weight,
        mcts_max_rollout_steps=mcts_max_rollout_steps,
        mcts_time_limit_seconds=mcts_time_limit_seconds,
    )
    app.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Ultimate Tic Tac Toe with pygame.")
    parser.add_argument(
        "--mode",
        choices=["human-human", "human-ai", "ai-ai"],
        default="human-ai",
        help="Initial game mode.",
    )
    parser.add_argument(
        "--ai",
        choices=["random", "mcts"],
        default="mcts",
        help="Initial computer agent.",
    )
    parser.add_argument(
        "--human-player",
        type=parse_player,
        default=PLAYER_X,
        help="Human side in human-ai mode.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mcts-iterations", type=int, default=1_000)
    parser.add_argument("--mcts-exploration-weight", type=float, default=2 ** 0.5)
    parser.add_argument("--mcts-max-rollout-steps", type=int, default=90)
    parser.add_argument("--mcts-time-limit", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_game(
        mode=args.mode,
        ai=args.ai,
        human_player=args.human_player,
        seed=args.seed,
        mcts_iterations=args.mcts_iterations,
        mcts_exploration_weight=args.mcts_exploration_weight,
        mcts_max_rollout_steps=args.mcts_max_rollout_steps,
        mcts_time_limit_seconds=args.mcts_time_limit,
    )


if __name__ == "__main__":
    main()

