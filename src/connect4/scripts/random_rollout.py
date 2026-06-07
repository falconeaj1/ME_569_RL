"""Run masked random actions against a fixed Connect Four opponent."""

from __future__ import annotations

import argparse

import numpy as np

from connect4.agents import agent_from_name
from connect4.core import Player
from connect4.env import ConnectFourVsAgentEnv


def run_rollouts(
    episodes: int,
    opponent_name: str,
    learner_player: Player,
    seed: int | None,
) -> None:
    rng = np.random.default_rng(seed)
    opponent = agent_from_name(opponent_name, seed=seed)
    env = ConnectFourVsAgentEnv(opponent=opponent, learner_player=learner_player)
    returns: list[float] = []

    for episode in range(1, episodes + 1):
        _, info = env.reset(seed=None if seed is None else seed + episode)
        done = False
        episode_return = 0.0
        steps = 0

        while not done:
            legal_actions = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal_actions))
            _, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated

        returns.append(episode_return)
        print(
            f"episode={episode} return={episode_return:.1f} "
            f"steps={steps} status={info['status'].value} winner={info['winner']}"
        )

    print(f"mean_return={float(np.mean(returns)):.3f}")


def parse_player(value: str) -> Player:
    normalized = value.strip().lower()
    if normalized in {"red", "r", "1"}:
        return Player.RED
    if normalized in {"yellow", "y", "-1"}:
        return Player.YELLOW
    raise argparse.ArgumentTypeError("player must be red or yellow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run masked random Connect Four rollouts.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--opponent",
        choices=["random", "bad", "heuristic", "good", "mcts", "first-legal"],
        default="random",
    )
    parser.add_argument("--learner-player", type=parse_player, default=Player.RED)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rollouts(
        episodes=args.episodes,
        opponent_name=args.opponent,
        learner_player=args.learner_player,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
