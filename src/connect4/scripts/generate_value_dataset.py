"""Generate Connect Four self-play histories for offline value learning."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from connect4.agents import Agent
from connect4.core import ConnectFourState, GameStatus, Player
from connect4.scripts.evaluate_agents import build_agent


DEFAULT_AGENT_MIX = "heuristic=0.15,mcts:100=0.30,mcts:500=0.35,mcts:1500=0.20"


@dataclass(frozen=True)
class GeneratedGame:
    game_index: int
    red_agent: str
    yellow_agent: str
    winner: str
    moves: int


def parse_agent_mix(value: str) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for part in value.split(","):
        if not part.strip():
            continue
        spec, weight_text = part.split("=", maxsplit=1)
        entries.append((spec.strip(), float(weight_text)))
    if not entries:
        raise argparse.ArgumentTypeError("agent mix must contain at least one spec=weight entry")
    total = sum(weight for _, weight in entries)
    if total <= 0:
        raise argparse.ArgumentTypeError("agent mix weights must sum to a positive value")
    return [(spec, weight / total) for spec, weight in entries]


def choose_agent_spec(agent_mix: list[tuple[str, float]], rng: np.random.Generator) -> str:
    specs = [spec for spec, _ in agent_mix]
    weights = np.array([weight for _, weight in agent_mix], dtype=np.float64)
    return str(rng.choice(specs, p=weights))


def play_labeled_game(
    game_index: int,
    red_spec: str,
    yellow_spec: str,
    red_agent: Agent,
    yellow_agent: Agent,
) -> tuple[GeneratedGame, list[np.ndarray], list[int], list[int], list[int], list[float], list[float]]:
    state = ConnectFourState.new()
    boards: list[np.ndarray] = []
    current_players: list[int] = []
    actions: list[int] = []
    move_indices: list[int] = []

    while not state.is_terminal:
        agent = red_agent if state.current_player == Player.RED else yellow_agent
        boards.append(state.board.copy())
        current_players.append(int(state.current_player))
        move_indices.append(state.move_count)
        action = agent.select_action(state.copy())
        actions.append(int(action))
        state.drop_piece(action)

    winner = state.winner
    red_value = 0.0 if state.status == GameStatus.TIE else 1.0 if winner == Player.RED else -1.0
    current_player_values = [red_value * player for player in current_players]
    red_values = [red_value for _ in current_players]
    record = GeneratedGame(
        game_index=game_index,
        red_agent=red_spec,
        yellow_agent=yellow_spec,
        winner="tie" if winner is None else winner.label.lower(),
        moves=state.move_count,
    )
    return record, boards, current_players, actions, move_indices, current_player_values, red_values


def generate_dataset(
    games: int,
    agent_mix: list[tuple[str, float]],
    seed: int,
    output: Path,
    metadata_output: Path,
    progress_every: int,
) -> None:
    rng = np.random.default_rng(seed)
    all_boards: list[np.ndarray] = []
    all_current_players: list[int] = []
    all_actions: list[int] = []
    all_game_indices: list[int] = []
    all_move_indices: list[int] = []
    all_values_current: list[float] = []
    all_values_red: list[float] = []
    game_records: list[GeneratedGame] = []

    start = time.perf_counter()
    for game_index in range(games):
        red_spec = choose_agent_spec(agent_mix, rng)
        yellow_spec = choose_agent_spec(agent_mix, rng)
        red_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        yellow_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        record, boards, current_players, actions, move_indices, values_current, values_red = play_labeled_game(
            game_index=game_index,
            red_spec=red_spec,
            yellow_spec=yellow_spec,
            red_agent=build_agent(red_spec, red_seed),
            yellow_agent=build_agent(yellow_spec, yellow_seed),
        )
        game_records.append(record)
        all_boards.extend(boards)
        all_current_players.extend(current_players)
        all_actions.extend(actions)
        all_game_indices.extend([game_index] * len(boards))
        all_move_indices.extend(move_indices)
        all_values_current.extend(values_current)
        all_values_red.extend(values_red)

        if progress_every and (game_index + 1) % progress_every == 0:
            print(f"{game_index + 1}/{games} games, {len(all_boards)} positions", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        boards=np.array(all_boards, dtype=np.int8),
        current_players=np.array(all_current_players, dtype=np.int8),
        actions=np.array(all_actions, dtype=np.int8),
        game_indices=np.array(all_game_indices, dtype=np.int32),
        move_indices=np.array(all_move_indices, dtype=np.int16),
        values_current_player=np.array(all_values_current, dtype=np.float32),
        values_red=np.array(all_values_red, dtype=np.float32),
    )

    elapsed = time.perf_counter() - start
    metadata = {
        "games": games,
        "positions": len(all_boards),
        "seed": seed,
        "agent_mix": [{"spec": spec, "weight": weight} for spec, weight in agent_mix],
        "elapsed_seconds": elapsed,
        "labels": {
            "values_current_player": "final game outcome from the side-to-move perspective at each stored state",
            "values_red": "final game outcome from Red perspective",
        },
        "games_detail": [asdict(record) for record in game_records],
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {output}")
    print(f"wrote {metadata_output}")
    print(f"generated {len(all_boards)} positions from {games} games in {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Connect Four value-network training data.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--agent-mix", type=parse_agent_mix, default=parse_agent_mix(DEFAULT_AGENT_MIX))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/value_data/connect4_value_dataset.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/value_data/connect4_value_dataset_metadata.json"))
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        games=args.games,
        agent_mix=args.agent_mix,
        seed=args.seed,
        output=args.output,
        metadata_output=args.metadata,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
