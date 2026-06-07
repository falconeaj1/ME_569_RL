import unittest

from connect4.mcts import MCTS
from connect4.tictactoe.tictactoe_mcts import EMPTY, PLAYER_O, PLAYER_X, TicTacToeState


class TicTacToeMCTSTests(unittest.TestCase):
    def test_tictactoe_detects_winner(self) -> None:
        state = TicTacToeState.from_rows(
            [
                [PLAYER_X, PLAYER_X, PLAYER_X],
                [PLAYER_O, PLAYER_O, EMPTY],
                [EMPTY, EMPTY, EMPTY],
            ],
            current_player=PLAYER_O,
        )

        self.assertTrue(state.is_terminal)
        self.assertEqual(state.winner, PLAYER_X)
        self.assertEqual(state.result_for(PLAYER_X), 1.0)
        self.assertEqual(state.result_for(PLAYER_O), -1.0)

    def test_mcts_selects_only_legal_action(self) -> None:
        state = TicTacToeState.from_rows(
            [
                [PLAYER_X, PLAYER_O, PLAYER_X],
                [PLAYER_X, PLAYER_O, PLAYER_O],
                [PLAYER_O, PLAYER_X, EMPTY],
            ],
            current_player=PLAYER_X,
        )

        result = MCTS(iterations=20, seed=0).search(state)

        self.assertEqual(result.action, 8)
        self.assertEqual(result.root_visits, 20)

    def test_mcts_time_limit_still_returns_legal_action(self) -> None:
        state = TicTacToeState()

        result = MCTS(iterations=100_000, time_limit_seconds=0.001, seed=0).search(state)

        self.assertIn(result.action, state.legal_actions())
        self.assertGreaterEqual(result.root_visits, len(state.legal_actions()))


if __name__ == "__main__":
    unittest.main()
