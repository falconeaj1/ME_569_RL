import unittest

from connect4.mcts import MCTS
from connect4.tutorials.tictactoe_mcts import EMPTY, O, X, TicTacToeState


class TicTacToeMCTSTests(unittest.TestCase):
    def test_tictactoe_detects_winner(self) -> None:
        state = TicTacToeState.from_rows(
            [
                [X, X, X],
                [O, O, EMPTY],
                [EMPTY, EMPTY, EMPTY],
            ],
            current_player=O,
        )

        self.assertTrue(state.is_terminal)
        self.assertEqual(state.winner, X)
        self.assertEqual(state.result_for(X), 1.0)
        self.assertEqual(state.result_for(O), -1.0)

    def test_mcts_selects_only_legal_action(self) -> None:
        state = TicTacToeState.from_rows(
            [
                [X, O, X],
                [X, O, O],
                [O, X, EMPTY],
            ],
            current_player=X,
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
