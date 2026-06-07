"""Small NumPy value network for Connect Four board evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connect4.core import ConnectFourState, Player


INPUT_SIZE = 42


def encode_state(state: ConnectFourState) -> np.ndarray:
    """Flatten a board from the current player's perspective."""

    return (state.board * int(state.current_player)).astype(np.float32).reshape(-1)


def encode_arrays(boards: np.ndarray, current_players: np.ndarray) -> np.ndarray:
    """Encode many boards from their side-to-move perspectives."""

    players = current_players.astype(np.float32).reshape(-1, 1, 1)
    return (boards.astype(np.float32) * players).reshape(boards.shape[0], -1)


@dataclass
class ValueNetwork:
    """One-hidden-layer tanh MLP trained with NumPy."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    metadata: dict[str, object]

    @classmethod
    def initialize(cls, hidden_size: int = 64, seed: int = 0) -> "ValueNetwork":
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 1.0 / np.sqrt(INPUT_SIZE), size=(INPUT_SIZE, hidden_size)).astype(np.float32)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), size=(hidden_size, 1)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2, metadata={"hidden_size": hidden_size, "seed": seed})

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        x2d = np.atleast_2d(x.astype(np.float32))
        hidden = np.tanh(x2d @ self.w1 + self.b1)
        output = np.tanh(hidden @ self.w2 + self.b2)
        return output.reshape(-1)

    def predict_state(self, state: ConnectFourState) -> float:
        return float(self.predict_features(encode_state(state))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            metadata=json.dumps(self.metadata, indent=2),
        )

    @classmethod
    def load(cls, path: Path) -> "ValueNetwork":
        with np.load(path, allow_pickle=False) as data:
            metadata_text = str(data["metadata"].item()) if "metadata" in data else "{}"
            return cls(
                w1=data["w1"].astype(np.float32),
                b1=data["b1"].astype(np.float32),
                w2=data["w2"].astype(np.float32),
                b2=data["b2"].astype(np.float32),
                metadata=json.loads(metadata_text),
            )


class ValueModelEvaluator:
    """Adapt a side-to-move value model to root-player MCTS values."""

    def __init__(self, model: ValueNetwork) -> None:
        self.model = model

    def __call__(self, state: ConnectFourState, root_player: int) -> float:
        current_player_value = self.model.predict_state(state)
        if int(state.current_player) == int(Player(root_player)):
            return current_player_value
        return -current_player_value
