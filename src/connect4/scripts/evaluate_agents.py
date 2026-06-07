"""Batch evaluations for Connect Four agents."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connect4.agents import Agent, BadHeuristicAgent, HeuristicAgent, MCTSAgent, RandomAgent
from connect4.core import ConnectFourState, GameStatus, Player


DEFAULT_MCTS_ITERATIONS = (1, 5, 10, 50, 100, 500, 1_000)
DEFAULT_MATCHUP_AGENTS = ("random", "bad", "heuristic", "mcts:100")


@dataclass(frozen=True)
class GameResult:
    winner: Player | None
    moves: int
    elapsed_seconds: float
    red_decision_seconds: float = 0.0
    yellow_decision_seconds: float = 0.0
    red_decisions: int = 0
    yellow_decisions: int = 0


@dataclass(frozen=True)
class TieSweepRow:
    iterations: int
    games: int
    ties: int
    red_wins: int
    yellow_wins: int
    tie_rate: float
    avg_moves: float
    elapsed_seconds: float


@dataclass(frozen=True)
class MatchupRow:
    red_agent: str
    yellow_agent: str
    games: int
    red_wins: int
    yellow_wins: int
    ties: int
    red_win_rate: float
    yellow_win_rate: float
    tie_rate: float
    avg_moves: float
    elapsed_seconds: float


@dataclass(frozen=True)
class VsHeuristicRow:
    experiment: str
    iterations: int
    exploration_weight: float
    rollout_policy: str
    value_model: str
    value_depth: int
    games_per_side: int
    total_games: int
    mcts_wins: int
    heuristic_wins: int
    ties: int
    mcts_win_rate: float
    mcts_win_ci95: float
    tie_rate: float
    avg_moves: float
    seconds_per_game: float
    seconds_per_move: float
    mcts_seconds_per_move: float
    simulations_per_mcts_move: int


def build_agent(spec: str, seed: int | None = None) -> Agent:
    """Build an agent from a compact spec like ``mcts:500`` or ``random``."""

    normalized = spec.strip().lower()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized in {"bad", "bad-heuristic"}:
        return BadHeuristicAgent(seed=seed)
    if normalized in {"heuristic", "good"}:
        return HeuristicAgent(seed=seed)
    if normalized in {"first", "first-legal"}:
        from connect4.agents import FirstLegalAgent

        return FirstLegalAgent()
    if normalized.startswith("mcts"):
        parts = normalized.split(":")
        iterations = int(parts[1]) if len(parts) >= 2 and parts[1] else 800
        time_limit = float(parts[2]) if len(parts) >= 3 and parts[2] else None
        return MCTSAgent(iterations=iterations, time_limit_seconds=time_limit, seed=seed)
    raise ValueError(f"unknown agent spec '{spec}'")


def build_mcts_agent(
    iterations: int,
    exploration_weight: float,
    rollout_policy: str = "random",
    value_model_path: Path | None = None,
    value_depth: int | None = None,
    seed: int | None = None,
) -> MCTSAgent:
    return MCTSAgent(
        iterations=iterations,
        exploration_weight=exploration_weight,
        rollout_policy=rollout_policy,
        value_model_path=None if value_model_path is None else str(value_model_path),
        value_depth=value_depth,
        seed=seed,
    )


def play_game(red_agent: Agent, yellow_agent: Agent) -> GameResult:
    """Play one Connect Four game between two agents."""

    state = ConnectFourState.new()
    start = time.perf_counter()
    red_decision_seconds = 0.0
    yellow_decision_seconds = 0.0
    red_decisions = 0
    yellow_decisions = 0

    while not state.is_terminal:
        agent = red_agent if state.current_player == Player.RED else yellow_agent
        decision_start = time.perf_counter()
        action = agent.select_action(state.copy())
        decision_elapsed = time.perf_counter() - decision_start
        if state.current_player == Player.RED:
            red_decision_seconds += decision_elapsed
            red_decisions += 1
        else:
            yellow_decision_seconds += decision_elapsed
            yellow_decisions += 1
        if not state.is_legal_action(action):
            winner = state.current_player.other
            return GameResult(
                winner=winner,
                moves=state.move_count,
                elapsed_seconds=time.perf_counter() - start,
                red_decision_seconds=red_decision_seconds,
                yellow_decision_seconds=yellow_decision_seconds,
                red_decisions=red_decisions,
                yellow_decisions=yellow_decisions,
            )
        state.drop_piece(action)

    winner = None if state.status == GameStatus.TIE else state.winner
    return GameResult(
        winner=winner,
        moves=state.move_count,
        elapsed_seconds=time.perf_counter() - start,
        red_decision_seconds=red_decision_seconds,
        yellow_decision_seconds=yellow_decision_seconds,
        red_decisions=red_decisions,
        yellow_decisions=yellow_decisions,
    )


def run_tie_sweep(
    iteration_counts: list[int],
    games: int,
    seed: int,
    progress_every: int,
) -> list[TieSweepRow]:
    rng = np.random.default_rng(seed)
    rows: list[TieSweepRow] = []

    for iterations in iteration_counts:
        if progress_every:
            print(f"running MCTS self-play iterations={iterations}, games={games}", flush=True)

        start = time.perf_counter()
        ties = 0
        red_wins = 0
        yellow_wins = 0
        moves: list[int] = []

        for game_index in range(1, games + 1):
            red_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            yellow_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            result = play_game(
                MCTSAgent(iterations=iterations, seed=red_seed),
                MCTSAgent(iterations=iterations, seed=yellow_seed),
            )
            moves.append(result.moves)
            if result.winner == Player.RED:
                red_wins += 1
            elif result.winner == Player.YELLOW:
                yellow_wins += 1
            else:
                ties += 1

            if progress_every and game_index % progress_every == 0:
                print(
                    f"  {game_index}/{games}: ties={ties}, red={red_wins}, yellow={yellow_wins}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start
        rows.append(
            TieSweepRow(
                iterations=iterations,
                games=games,
                ties=ties,
                red_wins=red_wins,
                yellow_wins=yellow_wins,
                tie_rate=ties / games,
                avg_moves=float(np.mean(moves)),
                elapsed_seconds=elapsed,
            )
        )
    return rows


def run_matchups(agent_specs: list[str], games: int, seed: int, progress_every: int) -> list[MatchupRow]:
    rng = np.random.default_rng(seed)
    rows: list[MatchupRow] = []

    for red_spec in agent_specs:
        for yellow_spec in agent_specs:
            if red_spec == yellow_spec:
                continue
            if progress_every:
                print(f"running {red_spec} red vs {yellow_spec} yellow, games={games}", flush=True)

            start = time.perf_counter()
            red_wins = 0
            yellow_wins = 0
            ties = 0
            moves: list[int] = []

            for game_index in range(1, games + 1):
                red_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                yellow_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                result = play_game(
                    build_agent(red_spec, red_seed),
                    build_agent(yellow_spec, yellow_seed),
                )
                moves.append(result.moves)
                if result.winner == Player.RED:
                    red_wins += 1
                elif result.winner == Player.YELLOW:
                    yellow_wins += 1
                else:
                    ties += 1

                if progress_every and game_index % progress_every == 0:
                    print(
                        f"  {game_index}/{games}: red={red_wins}, yellow={yellow_wins}, ties={ties}",
                        flush=True,
                    )

            elapsed = time.perf_counter() - start
            rows.append(
                MatchupRow(
                    red_agent=red_spec,
                    yellow_agent=yellow_spec,
                    games=games,
                    red_wins=red_wins,
                    yellow_wins=yellow_wins,
                    ties=ties,
                    red_win_rate=red_wins / games,
                    yellow_win_rate=yellow_wins / games,
                    tie_rate=ties / games,
                    avg_moves=float(np.mean(moves)),
                    elapsed_seconds=elapsed,
                )
            )
    return rows


def normal_ci95_half_width(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    p_hat = successes / trials
    return 1.96 * math.sqrt(p_hat * (1.0 - p_hat) / trials)


def run_vs_heuristic_condition(
    experiment: str,
    iterations: int,
    exploration_weight: float,
    rollout_policy: str,
    value_model_path: Path | None,
    value_depth: int | None,
    games_per_side: int,
    seed: int,
    progress_every: int,
) -> VsHeuristicRow:
    rng = np.random.default_rng(seed)
    mcts_wins = 0
    heuristic_wins = 0
    ties = 0
    moves: list[int] = []
    elapsed_seconds = 0.0
    all_decision_seconds = 0.0
    all_decisions = 0
    mcts_decision_seconds = 0.0
    mcts_decisions = 0

    pairings = [
        ("mcts-red", Player.RED),
        ("mcts-yellow", Player.YELLOW),
    ]
    for pairing_name, mcts_player in pairings:
        if progress_every:
            print(
                f"{experiment}: iter={iterations}, c={exploration_weight:g}, rollout={rollout_policy}, "
                f"{pairing_name}, games={games_per_side}",
                flush=True,
            )
        for game_index in range(1, games_per_side + 1):
            mcts_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            heuristic_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            mcts_agent = build_mcts_agent(
                iterations=iterations,
                exploration_weight=exploration_weight,
                rollout_policy=rollout_policy,
                value_model_path=value_model_path,
                value_depth=value_depth,
                seed=mcts_seed,
            )
            heuristic_agent = HeuristicAgent(seed=heuristic_seed)
            if mcts_player == Player.RED:
                result = play_game(mcts_agent, heuristic_agent)
                mcts_seconds = result.red_decision_seconds
                mcts_count = result.red_decisions
            else:
                result = play_game(heuristic_agent, mcts_agent)
                mcts_seconds = result.yellow_decision_seconds
                mcts_count = result.yellow_decisions

            moves.append(result.moves)
            elapsed_seconds += result.elapsed_seconds
            all_decision_seconds += result.red_decision_seconds + result.yellow_decision_seconds
            all_decisions += result.red_decisions + result.yellow_decisions
            mcts_decision_seconds += mcts_seconds
            mcts_decisions += mcts_count
            if result.winner is None:
                ties += 1
            elif result.winner == mcts_player:
                mcts_wins += 1
            else:
                heuristic_wins += 1

            if progress_every and game_index % progress_every == 0:
                print(
                    f"  {pairing_name} {game_index}/{games_per_side}: "
                    f"mcts={mcts_wins}, heuristic={heuristic_wins}, ties={ties}",
                    flush=True,
                )

    total_games = 2 * games_per_side
    value_model_label = "" if value_model_path is None else str(value_model_path)
    return VsHeuristicRow(
        experiment=experiment,
        iterations=iterations,
        exploration_weight=exploration_weight,
        rollout_policy=rollout_policy,
        value_model=value_model_label,
        value_depth=-1 if value_depth is None else value_depth,
        games_per_side=games_per_side,
        total_games=total_games,
        mcts_wins=mcts_wins,
        heuristic_wins=heuristic_wins,
        ties=ties,
        mcts_win_rate=mcts_wins / total_games,
        mcts_win_ci95=normal_ci95_half_width(mcts_wins, total_games),
        tie_rate=ties / total_games,
        avg_moves=float(np.mean(moves)),
        seconds_per_game=elapsed_seconds / total_games,
        seconds_per_move=all_decision_seconds / max(1, all_decisions),
        mcts_seconds_per_move=mcts_decision_seconds / max(1, mcts_decisions),
        simulations_per_mcts_move=iterations,
    )


def run_c_sweep(
    iterations_list: list[int],
    exploration_weights: list[float],
    games_per_side: int,
    seed: int,
    progress_every: int,
) -> list[VsHeuristicRow]:
    rows: list[VsHeuristicRow] = []
    condition_index = 0
    for iterations in iterations_list:
        for exploration_weight in exploration_weights:
            rows.append(
                run_vs_heuristic_condition(
                    experiment="c-sweep",
                    iterations=iterations,
                    exploration_weight=exploration_weight,
                    rollout_policy="random",
                    value_model_path=None,
                    value_depth=None,
                    games_per_side=games_per_side,
                    seed=seed + condition_index,
                    progress_every=progress_every,
                )
            )
            condition_index += 1
    return rows


def run_rollout_policy_sweep(
    iterations_list: list[int],
    rollout_policies: list[str],
    exploration_weight: float,
    games_per_side: int,
    seed: int,
    value_model_path: Path | None,
    value_depth: int,
    progress_every: int,
) -> list[VsHeuristicRow]:
    if "value" in rollout_policies and value_model_path is None:
        raise ValueError("rollout policy 'value' requires --value-model")
    rows: list[VsHeuristicRow] = []
    condition_index = 0
    for iterations in iterations_list:
        for policy in rollout_policies:
            model_path = value_model_path if policy == "value" else None
            depth = value_depth if policy == "value" else None
            rows.append(
                run_vs_heuristic_condition(
                    experiment="rollout-policy-sweep",
                    iterations=iterations,
                    exploration_weight=exploration_weight,
                    rollout_policy=policy,
                    value_model_path=model_path,
                    value_depth=depth,
                    games_per_side=games_per_side,
                    seed=seed + condition_index,
                    progress_every=progress_every,
                )
            )
            condition_index += 1
    return rows


def save_rows(rows: list[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_tie_sweep(rows: list[TieSweepRow], path: Path) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = path.parent / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row.iterations for row in rows], [100 * row.tie_rate for row in rows], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("MCTS iterations per move")
    ax.set_ylabel("Tie games (%)")
    ax.set_title("Connect Four MCTS self-play tie rate")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def plot_vs_heuristic(rows: list[VsHeuristicRow], path: Path, title: str) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = path.parent / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = sorted({row.iterations for row in rows})
    for iterations in groups:
        group_rows = [row for row in rows if row.iterations == iterations]
        if len({row.exploration_weight for row in group_rows}) > 1:
            x = [row.exploration_weight for row in group_rows]
            labels = None
            ax.set_xscale("log")
            ax.set_xlabel("UCT exploration constant c")
        else:
            x = list(range(len(group_rows)))
            labels = [row.rollout_policy for row in group_rows]
            ax.set_xlabel("Rollout policy")
        y = [100 * row.mcts_win_rate for row in group_rows]
        err = [100 * row.mcts_win_ci95 for row in group_rows]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=4, label=f"{iterations} simulations/move")
        if labels is not None:
            ax.set_xticks(x, labels)
    ax.set_ylabel("MCTS win rate vs heuristic (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def print_tie_sweep(rows: list[TieSweepRow]) -> None:
    header = "iterations  ties  red_wins  yellow_wins  tie_rate  avg_moves  seconds"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.iterations:>10}  {row.ties:>4}  {row.red_wins:>8}  {row.yellow_wins:>11}  "
            f"{row.tie_rate:>8.2%}  {row.avg_moves:>9.2f}  {row.elapsed_seconds:>7.2f}"
        )


def print_matchups(rows: list[MatchupRow]) -> None:
    header = "red_agent       yellow_agent    red_win  yellow_win  tie_rate  avg_moves  seconds"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.red_agent:<15} {row.yellow_agent:<15} {row.red_win_rate:>7.2%}  "
            f"{row.yellow_win_rate:>9.2%}  {row.tie_rate:>8.2%}  {row.avg_moves:>9.2f}  {row.elapsed_seconds:>7.2f}"
        )


def print_vs_heuristic(rows: list[VsHeuristicRow]) -> None:
    header = "experiment              iter       c  rollout    win_rate      ci  tie_rate  moves  mcts_s/m"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.experiment:<22} {row.iterations:>4}  {row.exploration_weight:>6.3g}  "
            f"{row.rollout_policy:<9} {row.mcts_win_rate:>8.2%}  ±{row.mcts_win_ci95:>5.2%}  "
            f"{row.tie_rate:>8.2%}  {row.avg_moves:>5.1f}  {row.mcts_seconds_per_move:>8.3f}"
        )


def parse_int_list(value: str) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return counts


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text in {"sqrt2", "sqrt(2)"}:
            values.append(math.sqrt(2))
        else:
            values.append(float(text))
    if not values:
        raise argparse.ArgumentTypeError("provide at least one float")
    return values


def parse_agent_list(value: str) -> list[str]:
    agents = [part.strip() for part in value.split(",") if part.strip()]
    if not agents:
        raise argparse.ArgumentTypeError("provide at least one agent spec")
    return agents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Connect Four agents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tie_parser = subparsers.add_parser("tie-sweep", help="Run MCTS self-play tie-rate sweep.")
    tie_parser.add_argument("--iterations", type=parse_int_list, default=list(DEFAULT_MCTS_ITERATIONS))
    tie_parser.add_argument("--games", type=int, default=20)
    tie_parser.add_argument("--seed", type=int, default=0)
    tie_parser.add_argument("--csv", type=Path, default=Path("outputs/connect4_mcts_tie_rates.csv"))
    tie_parser.add_argument("--plot", type=Path, default=Path("outputs/connect4_mcts_tie_rates.png"))
    tie_parser.add_argument("--no-plot", action="store_true")
    tie_parser.add_argument("--progress-every", type=int, default=5)

    matchup_parser = subparsers.add_parser("matchups", help="Run round-robin agent matchups.")
    matchup_parser.add_argument("--agents", type=parse_agent_list, default=list(DEFAULT_MATCHUP_AGENTS))
    matchup_parser.add_argument("--games", type=int, default=20)
    matchup_parser.add_argument("--seed", type=int, default=0)
    matchup_parser.add_argument("--csv", type=Path, default=Path("outputs/connect4_agent_matchups.csv"))
    matchup_parser.add_argument("--progress-every", type=int, default=5)

    c_parser = subparsers.add_parser("c-sweep", help="Sweep UCT exploration constants against heuristic.")
    c_parser.add_argument("--iterations", type=parse_int_list, default=[100, 500])
    c_parser.add_argument("--exploration-weights", type=parse_float_list, default=[0.25, 0.5, 1.0, math.sqrt(2), 2.0, 4.0])
    c_parser.add_argument("--games-per-side", type=int, default=20)
    c_parser.add_argument("--seed", type=int, default=0)
    c_parser.add_argument("--csv", type=Path, default=Path("outputs/connect4_c_sweep.csv"))
    c_parser.add_argument("--plot", type=Path, default=Path("outputs/connect4_c_sweep.png"))
    c_parser.add_argument("--no-plot", action="store_true")
    c_parser.add_argument("--progress-every", type=int, default=5)

    rollout_parser = subparsers.add_parser(
        "rollout-policy-sweep",
        help="Compare random, heuristic-guided, and value-cutoff rollouts against heuristic.",
    )
    rollout_parser.add_argument("--iterations", type=parse_int_list, default=[100, 500])
    rollout_parser.add_argument("--policies", type=parse_agent_list, default=["random", "heuristic"])
    rollout_parser.add_argument("--exploration-weight", type=float, default=math.sqrt(2))
    rollout_parser.add_argument("--games-per-side", type=int, default=20)
    rollout_parser.add_argument("--seed", type=int, default=0)
    rollout_parser.add_argument("--value-model", type=Path, default=None)
    rollout_parser.add_argument("--value-depth", type=int, default=8)
    rollout_parser.add_argument("--csv", type=Path, default=Path("outputs/connect4_rollout_policy_sweep.csv"))
    rollout_parser.add_argument("--plot", type=Path, default=Path("outputs/connect4_rollout_policy_sweep.png"))
    rollout_parser.add_argument("--no-plot", action="store_true")
    rollout_parser.add_argument("--progress-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tie-sweep":
        rows = run_tie_sweep(args.iterations, args.games, args.seed, args.progress_every)
        print_tie_sweep(rows)
        save_rows(rows, args.csv)
        print(f"\nwrote {args.csv}")
        if not args.no_plot:
            plot_tie_sweep(rows, args.plot)
            print(f"wrote {args.plot}")
    elif args.command == "matchups":
        rows = run_matchups(args.agents, args.games, args.seed, args.progress_every)
        print_matchups(rows)
        save_rows(rows, args.csv)
        print(f"\nwrote {args.csv}")
    elif args.command == "c-sweep":
        rows = run_c_sweep(args.iterations, args.exploration_weights, args.games_per_side, args.seed, args.progress_every)
        print_vs_heuristic(rows)
        save_rows(rows, args.csv)
        print(f"\nwrote {args.csv}")
        if not args.no_plot:
            plot_vs_heuristic(rows, args.plot, "UCT exploration constant sweep")
            print(f"wrote {args.plot}")
    elif args.command == "rollout-policy-sweep":
        rows = run_rollout_policy_sweep(
            iterations_list=args.iterations,
            rollout_policies=args.policies,
            exploration_weight=args.exploration_weight,
            games_per_side=args.games_per_side,
            seed=args.seed,
            value_model_path=args.value_model,
            value_depth=args.value_depth,
            progress_every=args.progress_every,
        )
        print_vs_heuristic(rows)
        save_rows(rows, args.csv)
        print(f"\nwrote {args.csv}")
        if not args.no_plot:
            plot_vs_heuristic(rows, args.plot, "MCTS rollout policy comparison")
            print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
