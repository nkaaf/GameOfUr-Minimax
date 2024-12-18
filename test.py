import dataclasses
import unittest
from typing import List

from minimax import MinimaxSimulation, State, NUM_OF_PIECES_PER_PLAYER, PLACE_ROSETTE_SAFE, \
    PLACE_START, PLACE_ROSETTE, PLACE_FINISH


class MinimaxTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sim = MinimaxSimulation()

        self.game_board_default = self.sim.game_board.copy()
        self.score_default = 0
        self.pieces_default = [PLACE_START] * NUM_OF_PIECES_PER_PLAYER

        self.state_default = State(self.game_board_default.copy(), self.score_default,
                                   self.score_default, self.pieces_default.copy(),
                                   self.pieces_default.copy(), 1, 2)

    def test0(self) -> None:
        """No movement"""
        piece_index = 0
        dice = 0

        current_state = self.state_default.copy()

        expected_state = self.state_default.copy()
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test1(self) -> None:
        """Piece is already in finish, no movement => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.pieces_1[piece_index] = PLACE_FINISH

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test2(self) -> None:
        """Piece cannot finish, dice is too high => No new state"""
        piece_index = 0
        dice = 3

        current_state = self.state_default.copy()
        current_state.game_board[5] += current_state.current_player
        current_state.pieces_1[piece_index] = 5

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test3(self) -> None:
        """Piece finishes"""
        piece_index = 0
        dice = 2

        current_state = self.state_default.copy()
        current_state.game_board[5] += current_state.current_player
        current_state.pieces_1[piece_index] = 5

        expected_state = self.state_default.copy()
        expected_state.pieces_1[piece_index] = PLACE_FINISH
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test4(self) -> None:
        """Piece cannot move, because same player is on this field => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.game_board[3] += current_state.current_player
        current_state.pieces_1[1] = 3

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test5(self) -> None:
        """Piece cannot be moved, because other player is on safe spot => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.game_board[9] += current_state.other_player
        current_state.pieces_2[0] = 9
        current_state.game_board[8] += current_state.current_player
        current_state.pieces_1[piece_index] = 8

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test6(self) -> None:
        """Normal move"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.game_board[6] += current_state.current_player
        current_state.pieces_1[piece_index] = 6

        expected_state = self.state_default.copy()
        expected_state.game_board[7] += current_state.current_player
        expected_state.pieces_1[piece_index] = 7
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test7(self) -> None:
        """Catch other player"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.game_board[7] += current_state.current_player
        current_state.pieces_1[piece_index] = 7
        current_state.game_board[8] += current_state.other_player
        current_state.pieces_2[0] = 8

        expected_state = self.state_default.copy()
        expected_state.game_board[8] += current_state.current_player
        expected_state.pieces_1[piece_index] = 8
        expected_state.pieces_2[0] = PLACE_START
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test8(self) -> None:
        """Move on Rosette field"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        current_state.game_board[1] += current_state.current_player
        current_state.pieces_1[piece_index] = 1

        expected_state = self.state_default.copy()
        expected_state.game_board[0] += current_state.current_player
        expected_state.pieces_1[piece_index] = 0
        expected_state.second_throw = True

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)
