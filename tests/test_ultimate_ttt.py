import unittest

import numpy as np

from connect4.mcts import MCTS
from connect4.ultimate_ttt import (
    LOCAL_TIE,
    PLAYER_O,
    PLAYER_X,
    UltimateStatus,
    UltimateTicTacToeState,
    row_col_to_action,
)


class UltimateTicTacToeStateTests(unittest.TestCase):
    def test_initial_position_allows_any_cell(self) -> None:
        state = UltimateTicTacToeState.new()

        self.assertEqual(len(state.legal_actions()), 81)
        self.assertIsNone(state.active_board)
        self.assertEqual(state.current_player, PLAYER_X)

    def test_move_sends_opponent_to_matching_local_board(self) -> None:
        state = UltimateTicTacToeState.new()

        next_state = state.next_state(row_col_to_action(4, 4))

        self.assertEqual(next_state.active_board, (1, 1))
        self.assertEqual(next_state.current_player, PLAYER_O)
        self.assertEqual(len(next_state.legal_actions()), 8)
        self.assertTrue(all(3 <= action // 9 <= 5 and 3 <= action % 9 <= 5 for action in next_state.legal_actions()))

    def test_completed_target_board_allows_any_open_board(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        local_status = np.zeros((3, 3), dtype=np.int8)
        local_status[1, 1] = LOCAL_TIE
        state = UltimateTicTacToeState.from_components(
            board=board,
            local_status=local_status,
            current_player=PLAYER_X,
            active_board=None,
        )

        next_state = state.next_state(row_col_to_action(1, 1))

        self.assertIsNone(next_state.active_board)
        self.assertNotIn(row_col_to_action(3, 3), next_state.legal_actions())
        self.assertEqual(len(next_state.legal_actions()), 71)

    def test_local_win_claims_sub_board(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[0, 0] = PLAYER_X
        board[0, 1] = PLAYER_X
        state = UltimateTicTacToeState.from_components(
            board=board,
            local_status=np.zeros((3, 3), dtype=np.int8),
            current_player=PLAYER_X,
            active_board=(0, 0),
            move_count=2,
        )

        next_state = state.next_state(row_col_to_action(0, 2))

        self.assertEqual(next_state.local_status[0, 0], PLAYER_X)
        self.assertEqual(next_state.status, UltimateStatus.ONGOING)
        self.assertEqual(next_state.active_board, (0, 2))

    def test_macro_win_ends_game(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[0, 6] = PLAYER_X
        board[0, 7] = PLAYER_X
        local_status = np.zeros((3, 3), dtype=np.int8)
        local_status[0, 0] = PLAYER_X
        local_status[0, 1] = PLAYER_X
        state = UltimateTicTacToeState.from_components(
            board=board,
            local_status=local_status,
            current_player=PLAYER_X,
            active_board=(0, 2),
            move_count=2,
        )

        next_state = state.next_state(row_col_to_action(0, 8))

        self.assertTrue(next_state.is_terminal)
        self.assertEqual(next_state.status, UltimateStatus.X_WIN)
        self.assertEqual(next_state.winner, PLAYER_X)
        self.assertEqual(next_state.result_for(PLAYER_X), 1.0)
        self.assertEqual(next_state.result_for(PLAYER_O), -1.0)

    def test_mcts_returns_legal_action(self) -> None:
        state = UltimateTicTacToeState.new().next_state(row_col_to_action(4, 4))

        result = MCTS(iterations=20, seed=0, max_rollout_steps=90).search(state)

        self.assertIn(result.action, state.legal_actions())
        self.assertEqual(result.root_visits, 20)


if __name__ == "__main__":
    unittest.main()
