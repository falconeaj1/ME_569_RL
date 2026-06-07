"""Gymnasium-compatible wrappers around :mod:`connect4.core`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from connect4.agents import Agent, RandomAgent
from connect4.core import (
    DEFAULT_COLS,
    DEFAULT_CONNECT_N,
    DEFAULT_ROWS,
    ConnectFourState,
    GameStatus,
    Player,
)


class ConnectFourEnv(gym.Env):
    """Alternating-turn environment.

    Each call to ``step`` applies the action for the current player, then the
    next observation is from the next player perspective by default. This is a
    useful base for self-play or algorithms that explicitly handle both sides.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        connect_n: int = DEFAULT_CONNECT_N,
        perspective_observation: bool = True,
        invalid_action_ends_episode: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.connect_n = connect_n
        self.perspective_observation = perspective_observation
        self.invalid_action_ends_episode = invalid_action_ends_episode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(cols)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(rows, cols),
            dtype=np.int8,
        )
        self.state = ConnectFourState.new(rows=rows, cols=cols, connect_n=connect_n)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        starting_player = Player(options.get("starting_player", Player.RED))
        self.state = ConnectFourState.new(
            rows=self.rows,
            cols=self.cols,
            connect_n=self.connect_n,
            starting_player=starting_player,
        )
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        player = self.state.current_player

        if not self.state.is_legal_action(int(action)):
            info = self._info()
            info["invalid_action"] = int(action)
            if self.invalid_action_ends_episode:
                return self._observation(), -1.0, True, False, info
            raise ValueError(f"invalid action {action}")

        self.state.drop_piece(int(action))
        terminated = self.state.is_terminal
        reward = self.state.result_for(player) if terminated else 0.0
        return self._observation(), reward, terminated, False, self._info()

    def render(self) -> str:
        return self.state.render_ascii()

    def _observation(self) -> np.ndarray:
        if self.perspective_observation:
            return self.state.observation(self.state.current_player)
        return self.state.observation()

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.state.action_mask(),
            "current_player": self.state.current_player,
            "status": self.state.status,
            "winner": self.state.winner,
            "last_move": self.state.last_move,
        }


class ConnectFourVsAgentEnv(gym.Env):
    """Single-agent environment where the opponent moves automatically."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        opponent: Agent | None = None,
        learner_player: Player = Player.RED,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        connect_n: int = DEFAULT_CONNECT_N,
        invalid_action_ends_episode: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.opponent = opponent or RandomAgent()
        self.learner_player = learner_player
        self.rows = rows
        self.cols = cols
        self.connect_n = connect_n
        self.invalid_action_ends_episode = invalid_action_ends_episode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(cols)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(rows, cols),
            dtype=np.int8,
        )
        self.state = ConnectFourState.new(rows=rows, cols=cols, connect_n=connect_n)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = ConnectFourState.new(rows=self.rows, cols=self.cols, connect_n=self.connect_n)
        if self.learner_player == Player.YELLOW:
            self._opponent_move()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state.current_player != self.learner_player:
            raise RuntimeError("environment is waiting for opponent, not learner")

        if not self.state.is_legal_action(int(action)):
            info = self._info()
            info["invalid_action"] = int(action)
            if self.invalid_action_ends_episode:
                return self._observation(), -1.0, True, False, info
            raise ValueError(f"invalid action {action}")

        self.state.drop_piece(int(action))
        if self.state.is_terminal:
            return self._terminal_step()

        self._opponent_move()
        if self.state.is_terminal:
            return self._terminal_step()

        return self._observation(), 0.0, False, False, self._info()

    def render(self) -> str:
        return self.state.render_ascii()

    def _opponent_move(self) -> None:
        action = self.opponent.select_action(self.state.copy())
        self.state.drop_piece(action)

    def _terminal_step(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = self.state.result_for(self.learner_player)
        if self.state.status == GameStatus.TIE:
            reward = 0.0
        return self._observation(), reward, True, False, self._info()

    def _observation(self) -> np.ndarray:
        return self.state.observation(self.learner_player)

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.state.action_mask(),
            "current_player": self.state.current_player,
            "learner_player": self.learner_player,
            "opponent": getattr(self.opponent, "name", type(self.opponent).__name__),
            "status": self.state.status,
            "winner": self.state.winner,
            "last_move": self.state.last_move,
        }
