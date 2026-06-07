import unittest

import numpy as np

from connect4.agents import HeuristicAgent
from connect4.core import ConnectFourState, GameStatus, Player
from connect4.env import ConnectFourEnv, ConnectFourVsAgentEnv
from connect4.mcts import MCTS


class ConnectFourCoreTests(unittest.TestCase):
    def test_vertical_win(self) -> None:
        state = ConnectFourState.new()
        for action in [0, 1, 0, 1, 0, 1, 0]:
            state.drop_piece(action)

        self.assertEqual(state.status, GameStatus.RED_WIN)
        self.assertEqual(state.winner, Player.RED)
        self.assertEqual(state.winning_cells, ((2, 0), (3, 0), (4, 0), (5, 0)))

    def test_diagonal_win(self) -> None:
        state = ConnectFourState.new()
        for action in [0, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3]:
            state.drop_piece(action)

        self.assertEqual(state.status, GameStatus.RED_WIN)
        self.assertEqual(state.winning_cells, ((5, 0), (4, 1), (3, 2), (2, 3)))

    def test_full_column_is_illegal(self) -> None:
        state = ConnectFourState.new()
        for _ in range(state.rows):
            state.drop_piece(0)

        self.assertFalse(state.is_legal_action(0))
        with self.assertRaises(ValueError):
            state.drop_piece(0)

    def test_observation_from_current_player_perspective(self) -> None:
        state = ConnectFourState.new()
        state.drop_piece(3)

        np.testing.assert_array_equal(state.observation(Player.YELLOW), -state.board)


class AgentAndEnvTests(unittest.TestCase):
    def test_heuristic_agent_takes_immediate_win(self) -> None:
        state = ConnectFourState.new()
        for action in [0, 6, 1, 6, 2, 5]:
            state.drop_piece(action)

        self.assertEqual(HeuristicAgent(seed=0).select_action(state), 3)

    def test_mcts_agent_finds_immediate_win(self) -> None:
        state = ConnectFourState.new()
        for action in [0, 6, 1, 6, 2, 5]:
            state.drop_piece(action)

        self.assertEqual(MCTS(iterations=100, seed=0).search(state).action, 3)

    def test_mcts_agent_blocks_immediate_loss(self) -> None:
        state = ConnectFourState.new()
        for action in [0, 1, 0, 1, 0]:
            state.drop_piece(action)

        self.assertEqual(MCTS(iterations=100, seed=0).search(state).action, 0)

    def test_alternating_env_returns_action_mask(self) -> None:
        env = ConnectFourEnv()
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, (6, 7))
        self.assertEqual(info["action_mask"].shape, (7,))
        _, reward, terminated, truncated, info = env.step(3)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["current_player"], Player.YELLOW)

    def test_vs_agent_env_steps_opponent_automatically(self) -> None:
        env = ConnectFourVsAgentEnv()
        observation, info = env.reset(seed=0)
        self.assertEqual(observation.shape, (6, 7))
        _, reward, terminated, _, info = env.step(3)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertEqual(info["current_player"], Player.RED)
        self.assertEqual(env.state.move_count, 2)


if __name__ == "__main__":
    unittest.main()
