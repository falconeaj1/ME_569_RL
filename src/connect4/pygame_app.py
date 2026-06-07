"""Interactive pygame frontend for Connect Four."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import pygame
import pygame.gfxdraw

from connect4.agents import Agent, agent_from_name, evaluate_board
from connect4.core import ConnectFourState, GameStatus, Player
from connect4.mcts import MCTSSearch


CELL_SIZE = 86
HEADER_HEIGHT = 86
EVAL_BAR_HEIGHT = 116
PANEL_WIDTH = 520
BOARD_BLUE = (30, 89, 184)
BACKGROUND = (22, 24, 29)
PANEL_BG = (32, 35, 42)
BUTTON_BG = (56, 61, 73)
BUTTON_ACTIVE = (78, 115, 178)
BUTTON_DISABLED = (43, 47, 56)
BUTTON_TEXT = (245, 245, 245)
BUTTON_DISABLED_TEXT = (120, 125, 136)
GRID_EMPTY = (245, 245, 245)
RED = (220, 50, 47)
YELLOW = (244, 196, 48)
WHITE = (245, 245, 245)
MUTED = (170, 174, 184)
WIN_HIGHLIGHT = (255, 255, 255)
WIN_GLOW = (40, 255, 180)
LAST_MOVE_HIGHLIGHT = (255, 255, 255)
LAST_MOVE_GLOW = (255, 232, 128)
EVAL_GREEN = (42, 220, 135)
EVAL_ORANGE = (255, 178, 66)
EVAL_NEUTRAL = (75, 80, 90)


AGENT_CHOICES = ["random", "bad", "heuristic", "mcts"]
MCTS_PRESETS = {
    "fast": {"iterations": 80, "time_limit": 0.25},
    "medium": {"iterations": 400, "time_limit": 1.0},
    "strong": {"iterations": 1_500, "time_limit": 2.5},
}
ITERATION_CHOICES = [40, 80, 200, 400, 800, 1_500, 3_000, 6_000]
TIME_CHOICES: list[float | None] = [None, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: Callable[[], None]
    active: Callable[[], bool] = lambda: False
    enabled: Callable[[], bool] = lambda: True


def draw_smooth_circle(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int,
) -> None:
    """Draw a filled anti-aliased circle."""

    x, y = center
    pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
    pygame.gfxdraw.aacircle(surface, x, y, radius, color)


def draw_smooth_ring(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int,
    width: int,
) -> None:
    """Draw an anti-aliased circular ring."""

    for offset in range(width):
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius - offset, color)


@dataclass(frozen=True)
class ColumnEvaluation:
    action: int
    heuristic_score: int
    mcts_visits: int = 0
    mcts_value: float = 0.0
    is_mcts_choice: bool = False


class ConnectFourPygameApp:
    """Pygame app with playable agents and optional move evaluation overlay."""

    def __init__(
        self,
        mode: str,
        ai: str,
        human_player: Player,
        seed: int | None,
        mcts_iterations: int,
        mcts_exploration_weight: float,
        mcts_max_rollout_steps: int,
        mcts_time_limit_seconds: float | None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Connect Four")

        self.state = ConnectFourState.new()
        self.history: list[ConnectFourState] = []
        self.mode = mode
        self.ai = ai
        self.human_player = human_player
        self.seed = seed
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_weight = mcts_exploration_weight
        self.mcts_max_rollout_steps = mcts_max_rollout_steps
        self.mcts_time_limit_seconds = mcts_time_limit_seconds
        self.mcts_profile = self._profile_from_settings(mcts_iterations, mcts_time_limit_seconds)
        self.eval_enabled = False
        self.eval_time_slice_seconds = 0.015
        self.eval_search_key: tuple[bytes, int] | None = None
        self.eval_search: MCTSSearch | None = None
        self.eval_heuristics: dict[int, int] = {}
        self.last_agent_move_at = 0.0
        self.last_agent_description = ""

        self.agents = self._build_agents()
        self.width = self.state.cols * CELL_SIZE + PANEL_WIDTH
        self.height = self.state.rows * CELL_SIZE + HEADER_HEIGHT + EVAL_BAR_HEIGHT
        self.board_width = self.state.cols * CELL_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 26, bold=True)
        self.medium_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 15)
        self.tiny_font = pygame.font.SysFont("Arial", 12)
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
            self.eval_enabled = not self.eval_enabled
            self._clear_eval_cache()
        elif key == pygame.K_m:
            self.mode = "human-ai" if self.mode != "human-ai" else "ai-ai"
            self._rebuild_agents_for_current_position()
        elif key == pygame.K_LEFTBRACKET:
            if self._custom_mcts_controls_enabled():
                self._step_iterations(-1)
        elif key == pygame.K_RIGHTBRACKET:
            if self._custom_mcts_controls_enabled():
                self._step_iterations(1)
        elif key == pygame.K_MINUS:
            if self._custom_mcts_controls_enabled():
                self._step_time_limit(-1)
        elif key == pygame.K_EQUALS:
            if self._custom_mcts_controls_enabled():
                self._step_time_limit(1)
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                if button.enabled():
                    button.action()
                return

        if pos[0] >= self.board_width or pos[1] < HEADER_HEIGHT + EVAL_BAR_HEIGHT:
            return
        if self.state.status != GameStatus.ONGOING:
            return
        if self._current_agent() is not None:
            return

        col = pos[0] // CELL_SIZE
        if self.state.is_legal_action(col):
            self._record_history()
            self.state.drop_piece(col)
            self.last_agent_move_at = time.monotonic()
            self._clear_eval_cache()

    def _advance_agent_if_needed(self) -> None:
        if self.state.status != GameStatus.ONGOING:
            return

        agent = self._current_agent()
        if agent is None:
            return
        if time.monotonic() - self.last_agent_move_at < 0.25:
            return

        start = time.perf_counter()
        action = agent.select_action(self.state.copy())
        elapsed = time.perf_counter() - start
        if self.state.is_legal_action(action):
            self._record_history()
            self.state.drop_piece(action)
            self.last_agent_description = f"{agent.name} played col {action} in {elapsed:.2f}s"
        self.last_agent_move_at = time.monotonic()
        self._clear_eval_cache()

    def _build_agents(self) -> dict[Player, Agent | None]:
        def build_ai(offset_seed: int | None = self.seed) -> Agent:
            return agent_from_name(
                self.ai,
                seed=offset_seed,
                mcts_iterations=self.mcts_iterations,
                mcts_exploration_weight=self.mcts_exploration_weight,
                mcts_max_rollout_steps=self.mcts_max_rollout_steps,
                mcts_time_limit_seconds=self.mcts_time_limit_seconds,
            )

        if self.mode == "human-human":
            return {Player.RED: None, Player.YELLOW: None}
        if self.mode == "human-ai":
            return {self.human_player: None, self.human_player.other: build_ai()}
        if self.mode == "ai-ai":
            return {
                Player.RED: build_ai(self.seed),
                Player.YELLOW: build_ai(None if self.seed is None else self.seed + 1),
            }
        raise ValueError(f"unknown mode '{self.mode}'")

    def _current_agent(self) -> Agent | None:
        return self.agents[self.state.current_player]

    def _reset(self) -> None:
        self.state = ConnectFourState.new()
        self.history = []
        self.agents = self._build_agents()
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
            while self.history and self._current_agent() is not None:
                self.state = self.history.pop()

        self.last_agent_move_at = time.monotonic()
        self.last_agent_description = ""
        self._clear_eval_cache()

    def _rebuild_agents_for_current_position(self) -> None:
        self.agents = self._build_agents()
        self.last_agent_move_at = time.monotonic()
        self.last_agent_description = ""
        self._clear_eval_cache()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._rebuild_agents_for_current_position()

    def _set_ai(self, name: str) -> None:
        self.ai = name
        self._rebuild_agents_for_current_position()

    def _set_human_player(self, player: Player) -> None:
        self.human_player = player
        self._rebuild_agents_for_current_position()

    def _set_mcts_preset(self, preset: str) -> None:
        settings = MCTS_PRESETS[preset]
        self.mcts_profile = preset
        self.mcts_iterations = int(settings["iterations"])
        self.mcts_time_limit_seconds = float(settings["time_limit"])
        self._rebuild_agents_for_current_position()

    def _set_mcts_custom(self) -> None:
        self.mcts_profile = "custom"
        self._rebuild_agents_for_current_position()

    def _step_iterations(self, direction: int) -> None:
        if not self._custom_mcts_controls_enabled():
            return
        current = self.mcts_iterations
        if current in ITERATION_CHOICES:
            index = ITERATION_CHOICES.index(current)
        else:
            nearest = min(range(len(ITERATION_CHOICES)), key=lambda i: abs(ITERATION_CHOICES[i] - current))
            index = nearest
        next_index = max(0, min(len(ITERATION_CHOICES) - 1, index + direction))
        self.mcts_iterations = ITERATION_CHOICES[next_index]
        self._rebuild_agents_for_current_position()

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

        index = TIME_CHOICES.index(current) if current in TIME_CHOICES else min(range(len(TIME_CHOICES)), key=lambda i: distance(TIME_CHOICES[i]))
        next_index = max(0, min(len(TIME_CHOICES) - 1, index + direction))
        self.mcts_time_limit_seconds = TIME_CHOICES[next_index]
        self._rebuild_agents_for_current_position()

    def _profile_from_settings(self, iterations: int, time_limit: float | None) -> str:
        for profile, settings in MCTS_PRESETS.items():
            if iterations == int(settings["iterations"]) and time_limit == float(settings["time_limit"]):
                return profile
        return "custom"

    def _ai_controls_enabled(self) -> bool:
        return self.mode != "human-human"

    def _human_player_controls_enabled(self) -> bool:
        return self.mode == "human-ai"

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
        self.eval_heuristics = {}

    def _layout_buttons(self) -> None:
        panel_x = self.board_width
        x = panel_x + 18
        y = 66
        full_width = PANEL_WIDTH - 36
        half_width = (full_width - 8) // 2
        height = 24
        row_gap = 5
        section_gap = 8

        def add_full(
            label: str,
            action: Callable[[], None],
            active: Callable[[], bool] = lambda: False,
            enabled: Callable[[], bool] = lambda: True,
        ) -> None:
            nonlocal y
            self.buttons.append(Button(label, pygame.Rect(x, y, full_width, height), action, active, enabled))
            y += height + row_gap

        def add_pair(
            left: tuple[str, Callable[[], None], Callable[[], bool], Callable[[], bool]],
            right: tuple[str, Callable[[], None], Callable[[], bool], Callable[[], bool]] | None = None,
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
            ("Undo (U)", self._undo, lambda: False, self._can_undo),
        )
        y += section_gap
        add_pair(
            ("Human-Human", lambda: self._set_mode("human-human"), lambda: self.mode == "human-human", lambda: True),
            ("Human-AI", lambda: self._set_mode("human-ai"), lambda: self.mode == "human-ai", lambda: True),
        )
        add_full("AI-AI", lambda: self._set_mode("ai-ai"), lambda: self.mode == "ai-ai")
        y += section_gap
        add_pair(
            (
                "Human red",
                lambda: self._set_human_player(Player.RED),
                lambda: self._human_player_controls_enabled() and self.human_player == Player.RED,
                self._human_player_controls_enabled,
            ),
            (
                "Human yellow",
                lambda: self._set_human_player(Player.YELLOW),
                lambda: self._human_player_controls_enabled() and self.human_player == Player.YELLOW,
                self._human_player_controls_enabled,
            ),
        )
        y += section_gap
        add_pair(
            (
                "AI random",
                lambda: self._set_ai("random"),
                lambda: self._ai_controls_enabled() and self.ai == "random",
                self._ai_controls_enabled,
            ),
            (
                "AI bad",
                lambda: self._set_ai("bad"),
                lambda: self._ai_controls_enabled() and self.ai == "bad",
                self._ai_controls_enabled,
            ),
        )
        add_pair(
            (
                "AI heuristic",
                lambda: self._set_ai("heuristic"),
                lambda: self._ai_controls_enabled() and self.ai == "heuristic",
                self._ai_controls_enabled,
            ),
            (
                "AI MCTS",
                lambda: self._set_ai("mcts"),
                lambda: self._ai_controls_enabled() and self.ai == "mcts",
                self._ai_controls_enabled,
            ),
        )
        y += section_gap
        add_pair(
            (
                "Fast 80/0.25s",
                lambda: self._set_mcts_preset("fast"),
                lambda: self._preset_is_active("fast"),
                self._mcts_controls_enabled,
            ),
            (
                "Medium 400/1s",
                lambda: self._set_mcts_preset("medium"),
                lambda: self._preset_is_active("medium"),
                self._mcts_controls_enabled,
            ),
        )
        add_pair(
            (
                "Strong 1500/2.5s",
                lambda: self._set_mcts_preset("strong"),
                lambda: self._preset_is_active("strong"),
                self._mcts_controls_enabled,
            ),
            (
                "Custom",
                self._set_mcts_custom,
                lambda: self._mcts_controls_enabled() and self.mcts_profile == "custom",
                self._mcts_controls_enabled,
            ),
        )
        add_pair(
            ("Iters -", lambda: self._step_iterations(-1), lambda: False, self._custom_mcts_controls_enabled),
            ("Iters +", lambda: self._step_iterations(1), lambda: False, self._custom_mcts_controls_enabled),
        )
        add_pair(
            ("Time -", lambda: self._step_time_limit(-1), lambda: False, self._custom_mcts_controls_enabled),
            ("Time +", lambda: self._step_time_limit(1), lambda: False, self._custom_mcts_controls_enabled),
        )
        y += section_gap
        add_full("Evaluator strip (E)", self._toggle_eval, lambda: self.eval_enabled)
        self.panel_details_y = y + 6

    def _preset_is_active(self, preset: str) -> bool:
        return self._mcts_controls_enabled() and self.mcts_profile == preset

    def _evaluations(self) -> dict[int, ColumnEvaluation]:
        if not self.eval_enabled or self.state.status != GameStatus.ONGOING:
            return {}

        key = (
            self.state.board.tobytes(),
            int(self.state.current_player),
        )
        if key != self.eval_search_key:
            player = self.state.current_player
            self.eval_heuristics = {}
            for action in self.state.legal_actions():
                next_state = self.state.next_state(action)
                self.eval_heuristics[action] = evaluate_board(next_state.board, player, self.state.connect_n)
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
        evals: dict[int, ColumnEvaluation] = {
            action: ColumnEvaluation(
                action=action,
                heuristic_score=score,
            )
            for action, score in self.eval_heuristics.items()
        }
        for action, stats in result.action_stats.items():
            if action in evals:
                evals[action] = ColumnEvaluation(
                    action=action,
                    heuristic_score=evals[action].heuristic_score,
                    mcts_visits=int(stats["visits"]),
                    mcts_value=float(stats["mean_value"]),
                    is_mcts_choice=action == result.action,
                )
        return evals

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND)
        evaluations = self._evaluations()
        self._draw_header()
        self._draw_evaluation_bar(evaluations)
        self._draw_board()
        self._draw_panel(evaluations)

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, BACKGROUND, pygame.Rect(0, 0, self.board_width, HEADER_HEIGHT))
        status = self._status_text()
        self.screen.blit(self.font.render(status, True, WHITE), (18, 12))
        help_text = "Click column to move. U undo. R reset. E evaluator. Q quit."
        self.screen.blit(self.small_font.render(help_text, True, MUTED), (18, 50))

    def _draw_board(self) -> None:
        board_top = HEADER_HEIGHT + EVAL_BAR_HEIGHT
        pygame.draw.rect(
            self.screen,
            BOARD_BLUE,
            pygame.Rect(0, board_top, self.board_width, self.state.rows * CELL_SIZE),
        )

        for row in range(self.state.rows):
            for col in range(self.state.cols):
                center = (
                    col * CELL_SIZE + CELL_SIZE // 2,
                    board_top + row * CELL_SIZE + CELL_SIZE // 2,
                )
                draw_smooth_circle(self.screen, cell_color(int(self.state.board[row, col])), center, CELL_SIZE // 2 - 8)

        self._draw_last_move_highlight(board_top)

        if self.state.winner is not None:
            self._draw_winning_highlight(board_top)

    def _draw_evaluation_bar(self, evaluations: dict[int, ColumnEvaluation]) -> None:
        rect = pygame.Rect(0, HEADER_HEIGHT, self.board_width, EVAL_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (18, 21, 27), rect)
        pygame.draw.line(self.screen, (39, 43, 52), rect.bottomleft, rect.bottomright, 2)

        if not self.eval_enabled:
            text = self.small_font.render("Evaluator off. Press E or use the side panel.", True, MUTED)
            self.screen.blit(text, (18, rect.y + 30))
            return
        if self.state.status != GameStatus.ONGOING:
            text = self.small_font.render("Game over. Reset to evaluate a new position.", True, MUTED)
            self.screen.blit(text, (18, rect.y + 30))
            return
        if not evaluations:
            text = self.small_font.render("Evaluator warming up...", True, MUTED)
            self.screen.blit(text, (18, rect.y + 30))
            return

        for col in range(self.state.cols):
            x = col * CELL_SIZE + 6
            card = pygame.Rect(x, rect.y + 14, CELL_SIZE - 12, EVAL_BAR_HEIGHT - 28)
            evaluation = evaluations.get(col)
            if evaluation is None:
                pygame.draw.rect(self.screen, (35, 38, 45), card, border_radius=5)
                self.screen.blit(self.tiny_font.render(f"C{col}", True, MUTED), (card.x + 6, card.y + 7))
                self.screen.blit(self.tiny_font.render("full", True, MUTED), (card.x + 6, card.y + 30))
                continue

            color = EVAL_GREEN if evaluation.is_mcts_choice else (28, 32, 39)
            pygame.draw.rect(self.screen, color, card, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, card, width=1, border_radius=5)
            lines = [
                f"C{col}",
                f"M {evaluation.mcts_value:+.2f}",
                f"N {evaluation.mcts_visits}",
            ]
            for line_index, line in enumerate(lines):
                text = self.tiny_font.render(line, True, WHITE)
                self.screen.blit(text, (card.x + 6, card.y + 7 + 16 * line_index))

    def _red_perspective_eval_score(self) -> float | None:
        if self.eval_search is None or self.eval_search.visits <= 0:
            return None
        player_to_move_value = self.eval_search.root.mean_value
        red_score = player_to_move_value * int(self.state.current_player)
        return max(-1.0, min(1.0, red_score))

    def _draw_winning_highlight(self, board_top: int) -> None:
        winning_cells = self.state.winning_cells
        if not winning_cells:
            return

        centers = [
            (
                col * CELL_SIZE + CELL_SIZE // 2,
                board_top + row * CELL_SIZE + CELL_SIZE // 2,
            )
            for row, col in winning_cells
        ]

        pygame.draw.lines(self.screen, WIN_GLOW, False, centers, 18)
        pygame.draw.lines(self.screen, WIN_HIGHLIGHT, False, centers, 8)

        for center in centers:
            draw_smooth_ring(self.screen, WIN_HIGHLIGHT, center, CELL_SIZE // 2 - 4, 4)
            draw_smooth_ring(self.screen, WIN_GLOW, center, CELL_SIZE // 2 + 1, 2)

    def _draw_last_move_highlight(self, board_top: int) -> None:
        if self.state.last_move is None:
            return

        row, col = self.state.last_move
        center = (
            col * CELL_SIZE + CELL_SIZE // 2,
            board_top + row * CELL_SIZE + CELL_SIZE // 2,
        )
        draw_smooth_ring(self.screen, LAST_MOVE_GLOW, center, CELL_SIZE // 2 - 2, 5)
        draw_smooth_ring(self.screen, LAST_MOVE_HIGHLIGHT, center, CELL_SIZE // 2 - 8, 3)
        pygame.draw.circle(self.screen, LAST_MOVE_HIGHLIGHT, center, 5)

    def _draw_panel(self, evaluations: dict[int, ColumnEvaluation]) -> None:
        panel_rect = pygame.Rect(self.board_width, 0, PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        self.screen.blit(self.font.render("Controls", True, WHITE), (self.board_width + 18, 16))

        for button in self.buttons:
            enabled = button.enabled()
            color = BUTTON_DISABLED if not enabled else BUTTON_ACTIVE if button.active() else BUTTON_BG
            pygame.draw.rect(self.screen, color, button.rect, border_radius=5)
            text_color = BUTTON_TEXT if enabled else BUTTON_DISABLED_TEXT
            label = self.small_font.render(button.label, True, text_color)
            self.screen.blit(label, (button.rect.x + 9, button.rect.y + 4))

        y = self.panel_details_y
        time_label = "off" if self.mcts_time_limit_seconds is None else f"{self.mcts_time_limit_seconds:g}s"
        details = [
            f"Mode: {self.mode}",
            f"AI: {self.ai if self._ai_controls_enabled() else 'off'}",
            f"MCTS profile: {self.mcts_profile if self._mcts_controls_enabled() else 'off'}",
            f"MCTS budget: {self.mcts_iterations} iters / {time_label}",
            f"Eval: {'on' if self.eval_enabled else 'off'}",
        ]

        for detail in details:
            self.screen.blit(self.small_font.render(detail, True, MUTED), (self.board_width + 18, y))
            y += 19

        if self.eval_enabled:
            y += 8
            self._draw_position_gauge(y, evaluations)

        if self.last_agent_description:
            clipped = self.last_agent_description[:44]
            self.screen.blit(self.tiny_font.render(clipped, True, WHITE), (self.board_width + 18, self.height - 22))

    def _draw_position_gauge(self, y: int, evaluations: dict[int, ColumnEvaluation]) -> int:
        label_x = self.board_width + 18
        gauge_x = label_x + 96
        gauge_y = y + 28
        gauge_height = 84
        gauge_width = 18
        red_score = self._red_perspective_eval_score()

        self.screen.blit(self.small_font.render("Position", True, WHITE), (label_x, y))
        visits = self.eval_search.visits if self.eval_search else 0
        self.screen.blit(self.tiny_font.render("Red", True, RED), (gauge_x - 3, gauge_y - 18))
        self.screen.blit(self.tiny_font.render("Yellow", True, YELLOW), (gauge_x - 10, gauge_y + gauge_height + 5))

        gauge_rect = pygame.Rect(gauge_x, gauge_y, gauge_width, gauge_height)
        midpoint = gauge_rect.centery
        pygame.draw.rect(self.screen, YELLOW, pygame.Rect(gauge_rect.x, midpoint, gauge_width, gauge_rect.bottom - midpoint))
        pygame.draw.rect(self.screen, RED, pygame.Rect(gauge_rect.x, gauge_rect.y, gauge_width, midpoint - gauge_rect.y))
        pygame.draw.rect(self.screen, WHITE, gauge_rect, width=1, border_radius=4)
        pygame.draw.line(self.screen, WHITE, (gauge_rect.x - 5, midpoint), (gauge_rect.right + 5, midpoint), 1)

        if red_score is None:
            marker_y = midpoint
            marker_label = "evaluating"
            marker_color = EVAL_NEUTRAL
        else:
            marker_y = gauge_rect.bottom - int(((red_score + 1.0) / 2.0) * gauge_height)
            marker_y = max(gauge_rect.y, min(gauge_rect.bottom, marker_y))
            if red_score > 0.05:
                marker_label = "Red ahead"
            elif red_score < -0.05:
                marker_label = "Yellow ahead"
            else:
                marker_label = "Even"
            marker_color = score_color(red_score)

        pygame.draw.circle(self.screen, WHITE, (gauge_rect.centerx, marker_y), 8)
        pygame.draw.circle(self.screen, marker_color, (gauge_rect.centerx, marker_y), 5)
        self.screen.blit(self.tiny_font.render(f"Visits: {visits}", True, MUTED), (gauge_x + 38, gauge_y + 12))
        self.screen.blit(self.small_font.render(marker_label, True, MUTED), (gauge_x + 38, gauge_y + 30))
        summary = self._evaluation_summary(evaluations)
        if summary:
            self.screen.blit(self.tiny_font.render(summary, True, MUTED), (gauge_x + 38, gauge_y + 52))
        return gauge_y + gauge_height + 19

    def _evaluation_summary(self, evaluations: dict[int, ColumnEvaluation]) -> str:
        if not evaluations:
            return ""
        total_visits = sum(evaluation.mcts_visits for evaluation in evaluations.values())
        if total_visits <= 0:
            return ""
        leader = max(evaluations.values(), key=lambda evaluation: evaluation.mcts_visits)
        share = leader.mcts_visits / total_visits
        stable_label = "stable" if total_visits >= 200 and share >= 0.85 else "searching"
        return f"Eval leader: col {leader.action} ({share:.0%}, {stable_label})"

    def _status_text(self) -> str:
        if self.state.status == GameStatus.TIE:
            return "Tie game"
        if self.state.winner is not None:
            return f"{self.state.winner.label} wins"

        agent = self._current_agent()
        actor = "Human" if agent is None else getattr(agent, "name", type(agent).__name__)
        return f"{self.state.current_player.label} to move: {actor}"


def cell_color(value: int) -> tuple[int, int, int]:
    if value == int(Player.RED):
        return RED
    if value == int(Player.YELLOW):
        return YELLOW
    return GRID_EMPTY


def score_color(red_score: float) -> tuple[int, int, int]:
    score = max(-1.0, min(1.0, red_score))
    if abs(score) < 0.05:
        return EVAL_NEUTRAL
    if score > 0:
        scale = 0.45 + 0.55 * score
        return (
            int(EVAL_NEUTRAL[0] * (1.0 - scale) + RED[0] * scale),
            int(EVAL_NEUTRAL[1] * (1.0 - scale) + RED[1] * scale),
            int(EVAL_NEUTRAL[2] * (1.0 - scale) + RED[2] * scale),
        )
    scale = 0.45 + 0.55 * abs(score)
    return (
        int(EVAL_NEUTRAL[0] * (1.0 - scale) + YELLOW[0] * scale),
        int(EVAL_NEUTRAL[1] * (1.0 - scale) + YELLOW[1] * scale),
        int(EVAL_NEUTRAL[2] * (1.0 - scale) + YELLOW[2] * scale),
    )


def run_game(
    mode: str = "human-human",
    ai: str = "random",
    human_player: Player = Player.RED,
    seed: int | None = None,
    mcts_iterations: int = 400,
    mcts_exploration_weight: float = 2 ** 0.5,
    mcts_max_rollout_steps: int = 1_000,
    mcts_time_limit_seconds: float | None = 1.0,
) -> None:
    app = ConnectFourPygameApp(
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


def parse_player(value: str) -> Player:
    normalized = value.strip().lower()
    if normalized in {"red", "r", "1"}:
        return Player.RED
    if normalized in {"yellow", "y", "-1"}:
        return Player.YELLOW
    raise argparse.ArgumentTypeError("player must be red or yellow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Connect Four with pygame.")
    parser.add_argument(
        "--mode",
        choices=["human-human", "human-ai", "ai-ai"],
        default="human-ai",
        help="Initial game mode.",
    )
    parser.add_argument(
        "--ai",
        choices=["random", "bad", "heuristic", "mcts", "first-legal"],
        default="mcts",
        help="Initial computer agent.",
    )
    parser.add_argument(
        "--human-player",
        type=parse_player,
        default=Player.RED,
        help="Human color in human-ai mode.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mcts-iterations", type=int, default=400)
    parser.add_argument("--mcts-exploration-weight", type=float, default=2 ** 0.5)
    parser.add_argument("--mcts-max-rollout-steps", type=int, default=1_000)
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
