"""Measure Tic Tac Toe MCTS self-play tie rate across iteration counts."""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connect4.mcts import MCTS
from connect4.tictactoe.tictactoe_mcts import PLAYER_O, PLAYER_X, TicTacToeState


DEFAULT_ITERATIONS = (1, 5, 10, 50, 100, 500, 1_000)


@dataclass(frozen=True)
class SweepRow:
    """Aggregated self-play result for one MCTS iteration count."""

    iterations: int
    games: int
    ties: int
    x_wins: int
    o_wins: int
    tie_rate: float
    avg_moves: float
    elapsed_seconds: float


def play_mcts_self_play_game(iterations: int, rng: np.random.Generator) -> tuple[int | None, int]:
    """Play one Tic Tac Toe game where both players choose moves with MCTS."""

    state = TicTacToeState()
    moves = 0

    while not state.is_terminal:
        search_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        result = MCTS(iterations=iterations, seed=search_seed).search(state)
        state = state.next_state(result.action)
        moves += 1

    return state.winner, moves


def run_tie_rate_sweep(
    iteration_counts: list[int] | tuple[int, ...] = DEFAULT_ITERATIONS,
    games_per_level: int = 100,
    seed: int = 0,
    progress_every: int = 0,
) -> list[SweepRow]:
    """Run MCTS self-play games and summarize tie rates by iteration count."""

    rng = np.random.default_rng(seed)
    rows: list[SweepRow] = []

    for iterations in iteration_counts:
        if progress_every:
            print(f"running iterations={iterations}, games={games_per_level}", flush=True)

        start = time.perf_counter()
        ties = 0
        x_wins = 0
        o_wins = 0
        move_counts: list[int] = []

        for game_index in range(1, games_per_level + 1):
            winner, moves = play_mcts_self_play_game(iterations, rng)
            move_counts.append(moves)
            if winner == PLAYER_X:
                x_wins += 1
            elif winner == PLAYER_O:
                o_wins += 1
            else:
                ties += 1

            if progress_every and game_index % progress_every == 0:
                print(
                    f"  completed {game_index}/{games_per_level} games "
                    f"(ties={ties}, x_wins={x_wins}, o_wins={o_wins})",
                    flush=True,
                )

        elapsed = time.perf_counter() - start
        rows.append(
            SweepRow(
                iterations=iterations,
                games=games_per_level,
                ties=ties,
                x_wins=x_wins,
                o_wins=o_wins,
                tie_rate=ties / games_per_level,
                avg_moves=float(np.mean(move_counts)),
                elapsed_seconds=elapsed,
            )
        )

        if progress_every:
            print(f"  finished iterations={iterations} in {elapsed:.2f}s\n", flush=True)

    return rows


def save_csv(rows: list[SweepRow], path: Path) -> None:
    """Write sweep results to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SweepRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_tie_rates(rows: list[SweepRow], path: Path | None = None) -> None:
    """Plot tie percentage against MCTS iterations per move."""

    if path is not None and "MPLCONFIGDIR" not in os.environ:
        cache_dir = path.parent / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib.pyplot as plt

    iterations = [row.iterations for row in rows]
    tie_percent = [100 * row.tie_rate for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, tie_percent, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("MCTS iterations per move")
    ax.set_ylabel("Tie games (%)")
    ax.set_title("Tic Tac Toe MCTS self-play tie rate")
    ax.set_ylim(-2, 102)
    ax.grid(True, which="both", alpha=0.3)

    for row in rows:
        ax.annotate(
            f"{100 * row.tie_rate:.0f}%",
            (row.iterations, 100 * row.tie_rate),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)


def print_summary(rows: list[SweepRow]) -> None:
    """Print a compact table of sweep results."""

    header = "iterations  ties  x_wins  o_wins  tie_rate  avg_moves  seconds"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.iterations:>10}  {row.ties:>4}  {row.x_wins:>6}  {row.o_wins:>6}  "
            f"{row.tie_rate:>8.2%}  {row.avg_moves:>9.2f}  {row.elapsed_seconds:>7.2f}"
        )


def parse_iteration_counts(value: str) -> list[int]:
    """Parse comma-separated positive integer iteration counts."""

    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("iteration counts must be positive integers")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Tic Tac Toe MCTS tie rates.")
    parser.add_argument(
        "--iterations",
        type=parse_iteration_counts,
        default=list(DEFAULT_ITERATIONS),
        help="Comma-separated MCTS iteration counts.",
    )
    parser.add_argument("--games", type=int, default=100, help="Games per iteration level.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("outputs/tictactoe_mcts_tie_rates.csv"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/tictactoe_mcts_tie_rates.png"))
    parser.add_argument("--no-plot", action="store_true", help="Skip writing a PNG plot.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N games. Use 0 to disable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_tie_rate_sweep(
        iteration_counts=args.iterations,
        games_per_level=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print_summary(rows)
    save_csv(rows, args.csv)
    print(f"\nwrote {args.csv}")

    if not args.no_plot:
        plot_tie_rates(rows, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
