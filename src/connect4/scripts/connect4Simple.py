"""Backward-compatible launcher for the pygame implementation.

The original prototype logic now lives in reusable modules under ``connect4``:

- ``connect4.core``: board state, legal moves, win/tie detection
- ``connect4.agents``: random, weak, heuristic, and MCTS agents
- ``connect4.env``: Gymnasium-style training environments
- ``connect4.pygame_app``: interactive pygame frontend
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from connect4.pygame_app import main


if __name__ == "__main__":
    main()
