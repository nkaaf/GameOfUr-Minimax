import unittest

import numpy

from minimax import MinimaxSimulation, State, PLACE_START, PLACE_FINISH, get_piece_mask


class MinimaxTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sim = MinimaxSimulation()

        self.score_default = 0
        self.pieces_default = numpy.uint32(PLACE_START)

        self.current_player_default = 1
        self.other_player_default = 2

        self.state_default = State(self.score_default, self.score_default, self.pieces_default.copy(),
                                   self.pieces_default.copy(), self.current_player_default, self.other_player_default)

    def test0(self) -> None:
        """No movement"""
        piece_index = 0
        dice = 0

        current_state = self.state_default.copy()

        expected_state = self.state_default.copy()
        expected_state.swap_player()
        expected_state.dice = dice
        expected_state.moved_piece = piece_index

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test1(self) -> None:
        """Piece is already in finish, no movement => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(PLACE_FINISH) << (4 * piece_index)

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test2(self) -> None:
        """Piece cannot finish, dice is too high => No new state"""
        piece_index = 0
        dice = 3

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(14) << (4 * piece_index)

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test3(self) -> None:
        """Piece finishes"""
        piece_index = 0
        dice = 2

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(14) << (4 * piece_index)

        expected_state = self.state_default.copy()
        expected_state.score_1 = 1
        expected_state.pieces_1 &= piece_mask_clear
        expected_state.pieces_1 |= numpy.uint32(PLACE_FINISH) << (4 * piece_index)
        expected_state.dice = dice
        expected_state.moved_piece = piece_index
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test4(self) -> None:
        """Piece cannot move, because same player is on this field => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index + 1)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(2) << (4 * (piece_index + 1))

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test5(self) -> None:
        """Piece cannot be moved, because other player is on safe spot => No new state"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(8) << (4 * piece_index)
        current_state.pieces_2 &= piece_mask_clear
        current_state.pieces_2 |= numpy.uint32(9) << (4 * piece_index)

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertIsNone(state_new)

    def test6(self) -> None:
        """Normal move"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(6) << (4 * piece_index)

        expected_state = self.state_default.copy()
        expected_state.pieces_1 &= piece_mask_clear
        expected_state.pieces_1 |= numpy.uint32(7) << (4 * piece_index)
        expected_state.dice = dice
        expected_state.moved_piece = piece_index
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test7(self) -> None:
        """Catch other player"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(7) << (4 * piece_index)
        current_state.pieces_2 &= piece_mask_clear
        current_state.pieces_2 |= numpy.uint32(8) << (4 * piece_index)

        expected_state = self.state_default.copy()
        expected_state.pieces_1 &= piece_mask_clear
        expected_state.pieces_1 |= numpy.uint32(8) << (4 * piece_index)
        expected_state.pieces_2 &= piece_mask_clear
        expected_state.pieces_2 |= numpy.uint32(PLACE_START) << (4 * piece_index)
        expected_state.dice = dice
        expected_state.moved_piece = piece_index
        expected_state.swap_player()

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)

    def test8(self) -> None:
        """Move on Rosette field"""
        piece_index = 0
        dice = 1

        current_state = self.state_default.copy()
        piece_mask_clear = ~get_piece_mask(piece_index)
        current_state.pieces_1 &= piece_mask_clear
        current_state.pieces_1 |= numpy.uint32(5) << (4 * piece_index)

        expected_state = self.state_default.copy()
        expected_state.pieces_1 &= piece_mask_clear
        expected_state.pieces_1 |= numpy.uint32(6) << (4 * piece_index)
        expected_state.dice = dice
        expected_state.moved_piece = piece_index
        expected_state.second_throw = True

        state_new = self.sim.simulate_step(current_state, piece_index, dice)

        self.assertEqual(expected_state, state_new)


class ArithmeticTest(unittest.TestCase):

    def test0(self) -> None:
        state = State(0, 0, numpy.uint32(0xA4FDA), numpy.uint32(0xD), 1, 2)

        state.piece_move(1, 0, numpy.uint8(0x9))
        self.assertEqual(numpy.uint32(0xA4FD9), state.pieces_1)

        state.piece_move(1, 1, numpy.uint8(0x9))
        self.assertEqual(numpy.uint32(0xA4F99), state.pieces_1)

        state.piece_move(1, 2, numpy.uint8(0x9))
        self.assertEqual(numpy.uint32(0xA4999), state.pieces_1)

        state.piece_move(1, 3, numpy.uint8(0x9))
        self.assertEqual(numpy.uint32(0xA9999), state.pieces_1)

        state.piece_move(1, 4, numpy.uint8(0x9))
        self.assertEqual(numpy.uint32(0x99999), state.pieces_1)
