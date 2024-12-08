import dataclasses
import unittest
from typing import List
from unittest import mock

from minimax import MinimaxSimulation, State, NUM_OF_PIECES_PER_PLAYER

ROSETTE = MinimaxSimulation.PLACE_ROSETTE
ROSETTE_SAFE = MinimaxSimulation.PLACE_ROSETTE_SAFE

START = MinimaxSimulation.PLACE_START
FINISH = MinimaxSimulation.PLACE_FINISH


class MinimaxTest(unittest.TestCase):

    def setUp(self) -> None:
        self.game_board_default = [ROSETTE, 0, 0, 0, ROSETTE, 0, 0, 0, 0, ROSETTE_SAFE, 0, 0, 0, 0,
                                   ROSETTE, 0, 0, 0, ROSETTE, 0]
        self.sim = MinimaxSimulation()

        self.places_default = [START] * NUM_OF_PIECES_PER_PLAYER

        self.score_default = 0
        self.step_default = 0
        self.state_default = self.create_state(self.game_board_default.copy(), self.score_default,
                                               self.score_default, self.places_default.copy(),
                                               self.places_default.copy(), self.step_default)

    @staticmethod
    def create_state(game_board: List[int], score_1: int, score_2: int, places_1: List[int],
                     places_2: List[int], step: int) -> State:
        return State(game_board, score_1, score_2, places_1, places_2, step)

    def add_state(self, state: State) -> State:
        return self.sim.state_list.add_new_state(state)

    def get_simulated_state(self, current_state: State) -> State:
        return self.sim.state_list.get_next_child(current_state)

    def test0(self):
        """Player 1 and Player 2 are in home. Player 1 throw 0"""
        current_player = 1
        other_player = 2
        dice = 0

        expected_place = START

        current_state = self.add_state(self.state_default)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step = self.get_simulated_state(current_state)

        self.assertEqual(simulated_step.game_board, self.game_board_default)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_1_new = self.places_default.copy()
        places_1_new[0] = expected_place
        self.assertEqual(simulated_step.places_1, places_1_new)
        self.assertEqual(simulated_step.places_2, self.places_default)

        self.assertEqual(len(current_state.children), 1)

    def test1(self):
        """Player 1 and Player 2 are in home. Player 1 goes on game_board"""
        current_player = 1
        other_player = 2
        dice = 1

        expected_place = 3

        current_state = self.add_state(self.state_default)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step = self.get_simulated_state(current_state)

        self.assertEqual(
            simulated_step.game_board[expected_place] - self.game_board_default[expected_place],
            current_player)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_1_new = self.places_default.copy()
        places_1_new[0] = expected_place
        self.assertEqual(simulated_step.places_1, places_1_new)
        self.assertEqual(simulated_step.places_2, self.places_default)

        changed_board = simulated_step.game_board.copy()
        changed_board[expected_place] -= current_player
        self.assertEqual(changed_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)

    def test2(self):
        """Player 1 and Player 2 are in home. Player 2 goes on first Rosette and throw the dice again."""
        MinimaxSimulation.throw_dices = mock.Mock(return_value=[3])

        current_player = 2
        other_player = 1
        dice = 4

        expected_place_first = 14

        current_state = self.add_state(self.state_default)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step = self.get_simulated_state(current_state)

        # Evaluation first throw

        self.assertEqual(simulated_step.game_board[expected_place_first] - self.game_board_default[
            expected_place_first], current_player)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_2_new = self.places_default.copy()
        places_2_new[0] = expected_place_first
        self.assertEqual(simulated_step.places_1, self.places_default)
        self.assertEqual(simulated_step.places_2, places_2_new)

        changed_board = simulated_step.game_board.copy()
        changed_board[expected_place_first] -= current_player
        self.assertEqual(changed_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)

        # Evaluation second throw

        expected_place_second = 8

        simulated_step = self.get_simulated_state(simulated_step)

        self.assertEqual(simulated_step.game_board[expected_place_second] - self.game_board_default[
            expected_place_second], current_player)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_2_new = self.places_default.copy()
        places_2_new[0] = expected_place_second
        self.assertEqual(simulated_step.places_1, self.places_default)
        self.assertEqual(simulated_step.places_2, places_2_new)

        changed_board = simulated_step.game_board.copy()
        changed_board[expected_place_second] -= current_player
        self.assertEqual(changed_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)

    def test3(self):
        """Player 1 has one piece which goes directly into finish."""
        current_player = 1
        other_player = 2
        dice = 2

        expected_place = FINISH

        state = dataclasses.replace(self.state_default)
        state.game_board[5] += current_player
        state.places_1[0] = 5

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step = self.get_simulated_state(current_state)

        self.assertEqual(simulated_step.game_board[5], self.game_board_default[5])
        self.assertEqual(simulated_step.score_1, 1)
        self.assertEqual(simulated_step.score_2, 0)

        places_1_new = self.places_default.copy()
        places_1_new[0] = expected_place
        self.assertEqual(simulated_step.places_1, places_1_new)
        self.assertEqual(simulated_step.places_2, self.places_default)

        self.assertEqual(simulated_step.game_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)

    def test4(self):
        """Player 1 has one piece which would go one step further than finish."""
        current_player = 1
        other_player = 2
        dice = 3

        state = dataclasses.replace(self.state_default)
        state.game_board[5] += current_player
        state.places_1[0] = 5

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        self.get_simulated_state(current_state)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER - 1)

    def test5(self):
        """Player 1 has one piece which is already in finish."""
        current_player = 1
        other_player = 2
        dice = 3

        state = dataclasses.replace(self.state_default)
        state.places_1[0] = FINISH

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        self.get_simulated_state(current_state)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER - 1)

    def test6(self):
        """Player 1 tries to go on a field, where a piece of itself is laying."""
        current_player = 1
        other_player = 2
        dice = 1

        state = dataclasses.replace(self.state_default)
        state.game_board[1] += current_player
        state.game_board[2] += current_player
        state.places_1[0] = 2
        state.places_1[1] = 1

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        self.get_simulated_state(current_state)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER - 1)

    def test7(self):
        """Player 1 tries to go on the middle Rosette, where a piece of player 2 is laying."""
        current_player = 1
        other_player = 2
        dice = 1

        state = dataclasses.replace(self.state_default)
        state.game_board[8] += current_player
        state.game_board[9] += other_player
        state.places_1[0] = 8
        state.places_2[0] = 9

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        self.get_simulated_state(current_state)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER - 1)

    def test8(self):
        """Player 1 goes on middle rosette and is not allowed to throw again."""
        current_player = 1
        other_player = 2
        dice = 1

        state = dataclasses.replace(self.state_default)
        state.game_board[8] += current_player
        state.places_1[0] = 8

        expected_place = 9

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step = self.get_simulated_state(current_state)

        self.assertEqual(
            simulated_step.game_board[expected_place] - self.game_board_default[expected_place],
            current_player)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_1_new = self.places_default.copy()
        places_1_new[0] = expected_place
        self.assertEqual(simulated_step.places_1, places_1_new)
        self.assertEqual(simulated_step.places_2, self.places_default)

        changed_board = simulated_step.game_board.copy()
        changed_board[expected_place] -= current_player
        self.assertEqual(changed_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)

    def test9(self):
        """Player 1 catches a piece of player 2."""
        current_player = 1
        other_player = 2
        dice = 1

        state = dataclasses.replace(self.state_default)
        state.game_board[7] += current_player
        state.game_board[8] += other_player
        state.places_1[0] = 7
        state.places_2[0] = 8

        expected_place_1 = 8
        expected_place_2 = START

        current_state = self.add_state(state)
        self.sim.simulate_step(current_player, other_player, current_state, dice, self.step_default)
        simulated_step =self.get_simulated_state(current_state)

        self.assertEqual(
            simulated_step.game_board[expected_place_1] - self.game_board_default[expected_place_1],
            current_player)
        self.assertEqual(simulated_step.score_1, 0)
        self.assertEqual(simulated_step.score_2, 0)

        places_1_new = self.places_default.copy()
        places_1_new[0] = expected_place_1
        self.assertEqual(simulated_step.places_1, places_1_new)
        places_2_new = self.places_default.copy()
        places_2_new[0] = expected_place_2
        self.assertEqual(simulated_step.places_2, places_2_new)

        changed_board = simulated_step.game_board.copy()
        changed_board[8] -= current_player
        self.assertEqual(changed_board, self.game_board_default)

        self.assertEqual(len(current_state.children), NUM_OF_PIECES_PER_PLAYER)
