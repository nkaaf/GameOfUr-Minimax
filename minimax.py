import sys
from dataclasses import dataclass, field
from typing import Optional, List

import graphviz
import numpy


class ListIndexSafe(list):
    def index_safe(self, *args, **kwargs) -> int:
        try:
            return self.index(*args, **kwargs)
        except ValueError:
            return -1


NUM_OF_PIECES_PER_PLAYER = 5
STEPS_IN_FUTURE = 8
PLAYER_1_MIN = True
VISUALIZE = False


@dataclass
class State:
    game_board: List[int]
    score_1: int
    score_2: int
    places_1: List[int]
    places_2: List[int]
    step: int
    parent_pos: Optional[int] = field(default=None)
    pos: int = field(init=False)
    children: List[int] = field(init=False, default_factory=list)
    child_iter: int = field(init=False, default=-1)


class StateList:
    def __init__(self):
        self.states: List[State] = []

    def __iter__(self):
        return iter(self.states)

    def add_new_state(self, state: State) -> State:
        state.pos = len(self.states)
        self.states.append(state)
        return state

    def get_parent(self, state: State) -> Optional[State]:
        if state.parent_pos is None:
            return None

        assert 0 <= state.parent_pos <= len(self.states), "State's parent position is not valid!"
        return self.states[state.parent_pos]

    def get_next_child_of_parent(self, state: State) -> Optional[State]:
        parent = self.get_parent(state)
        if parent is None:
            return None

        return self.get_next_child(parent)

    def get_next_child(self, state: State) -> Optional[State]:
        state.child_iter += 1
        try:
            return self.states[state.children[state.child_iter]]
        except IndexError:
            return None


class MinimaxSimulation:
    PLACE_ROSETTE = 3
    PLACE_START = -1
    PLACE_FINISH = -2

    def __init__(self) -> None:
        # Rules: https://www.mastersofgames.com/rules/royal-ur-rules.htm
        # Rule clarification: No safe spot. Second throw on every rosette.
        # States:
        # 0 -> No player on this field
        # 1 -> Player1 on this field
        # 2 -> Player2 on this field
        # 3 -> Rosette (another throw)
        # 4 = 3 + 1 -> Player 1 on rosette
        # 5 = 3 + 2 -> Player 2 on rosette
        self.game_board = [self.PLACE_ROSETTE, 0, 0, 0, self.PLACE_ROSETTE, 0, 0, 0, 0,
                           self.PLACE_ROSETTE, 0, 0, 0, 0, self.PLACE_ROSETTE, 0, 0, 0,
                           self.PLACE_ROSETTE, 0]

        # Indices of game_board path for both players
        self.path_1 = ListIndexSafe([3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 5, 4])
        self.path_2 = ListIndexSafe([17, 16, 15, 14, 6, 7, 8, 9, 10, 11, 12, 13, 19, 18])
        self.paths = self.player_based_list(self.path_1, self.path_2)

        # Version1: Calculate steps before the simulation
        self.dices_1 = self.throw_dices(STEPS_IN_FUTURE // 2)
        self.dices_2 = self.throw_dices(STEPS_IN_FUTURE // 2)
        self.dices = self.player_based_list(self.dices_1, self.dices_2)

        self.state_list = StateList()

        # ----- Start State ----- #

        # Indices of game_board places for both players
        # -1 -> Start
        # -2 -> Finish
        places_1 = places_2 = [self.PLACE_START] * NUM_OF_PIECES_PER_PLAYER
        # Number of pieces in finish for both players
        score_1 = score_2 = 0
        # Start state
        state = self.game_board

        self.start_state = self.state_list.add_new_state(
            State(state, score_1, score_2, places_1, places_2, 0))

    @staticmethod
    def throw_dices(n: int) -> numpy.ndarray[int]:
        # Throw probability: http://matthewdeutsch.com/projects/game-of-ur/
        return numpy.random.choice(range(4 + 1), n, p=[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])

    @staticmethod
    def player_based_list(e1, e2) -> list:
        return [None, e1, e2]

    def simulate_step(self, current_player: int, other_player: int, current_state: State, dice: int,
                      step: int) -> None:

        print(f"Position: {current_state.pos} - Step: {step}")
        print(f"Current player: {current_player}, other_player: {other_player}")
        print(f"Board state: {current_state.game_board} - Dice: {dice}")
        print(f"Places 1: {current_state.places_1} - Places 2: {current_state.places_2}")
        print(f"Score 1: {current_state.score_1} - Score 2: {current_state.score_2}")

        scores = self.player_based_list(current_state.score_1, current_state.score_2)
        game_board_current = current_state.game_board
        places_current_player = \
            self.player_based_list(current_state.places_1, current_state.places_2)[current_player]

        # ----- Check Win Condition ----- #

        if scores[current_player] == NUM_OF_PIECES_PER_PLAYER:
            print("Win - Keine Ahnung was jetzt")
            sys.exit(0)

        # ----- New State variables ----- #

        step_new = step + 1
        parent_pos = current_state.pos

        # ----- Check Dice == 0 ----- #

        if dice == 0:
            # No movement

            scores_new = scores.copy()
            game_board_new = game_board_current.copy()
            places_new = self.player_based_list(current_state.places_1.copy(),
                                                current_state.places_2.copy())

            state_new = State(game_board_new, scores_new[1], scores_new[2], places_new[1],
                              places_new[2], step_new, parent_pos)
            state_new = self.state_list.add_new_state(state_new)
            current_state.children = [state_new.pos]

            return

        # ----- Path variable ----- #

        path: ListIndexSafe[int] = self.paths[current_player]

        # ----- Creation of moves for every piece ----- #

        for piece_index, place_current in enumerate(places_current_player):

            # ----- New State variables ----- #

            scores_new = scores.copy()
            game_board_new = game_board_current.copy()
            places_new = self.player_based_list(current_state.places_1.copy(),
                                                current_state.places_2.copy())

            if place_current == self.PLACE_FINISH:
                # Piece is in finish

                continue

            # index of next path position in path list
            place_new_path_index = path.index_safe(place_current) + dice
            if place_new_path_index >= len(path):
                # Move cannot be done, because the piece has to be finished perfectly

                continue

            second_throw = False
            if place_new_path_index == len(path):
                # Piece is in finish with next move

                scores_new[current_player] += 1

                # Piece moves from current place to finish
                game_board_new[place_current] -= current_player
                places_new[current_player][piece_index] = self.PLACE_FINISH
            else:
                # Piece has not finished with next move

                # ID of next place
                place_new = path[place_new_path_index]
                # Current value of the new place
                place_new_current_value = game_board_current[place_new]

                if place_new_current_value - current_player in [0, self.PLACE_ROSETTE]:
                    # Move cannot be done, on the field is already a piece of the current_player,

                    continue

                if place_new_current_value - other_player == 0:
                    # Other player will be caught and returned to start

                    # current player moves from current place to new place
                    game_board_new[place_current] -= current_player
                    game_board_new[place_new] += current_player
                    places_new[current_player][piece_index] = place_new

                    # other player piece on new place moves to start
                    game_board_new[place_new] -= other_player
                    places_new[other_player][
                        places_new[other_player].index(place_new)] = self.PLACE_START

                elif place_new_current_value in [0, self.PLACE_ROSETTE]:
                    # Field is free

                    if place_new_current_value == self.PLACE_ROSETTE:
                        second_throw = True

                    if place_current != self.PLACE_START:
                        # Piece is already in the game

                        game_board_new[place_current] = 0

                    game_board_new[place_new] += current_player
                    places_new[current_player][piece_index] = place_new

            # ----- New State creation ----- #

            state_new = State(game_board_new, scores_new[1], scores_new[2], places_new[1],
                              places_new[2], step_new, parent_pos)
            state_new = self.state_list.add_new_state(state_new)
            current_state.children.append(state_new.pos)

            # ----- Second throw if on Rosette ----- #

            if second_throw:
                dice_second = self.throw_dices(1)[0]

                self.simulate_step(current_player, other_player, state_new, dice_second, step + 1)

        print()

    def visualize(self) -> None:
        graph = graphviz.Graph()

        for state in self.state_list:
            graph.node(str(state.pos), str(state.pos))

        for state in self.state_list:
            for child in state.children:
                graph.edge(str(state.pos), str(child))

        graph.view()

    def start(self) -> None:
        assert STEPS_IN_FUTURE % 2 == 0, "STEPS_IN_FUTURE must be even"

        assert len(self.dices_1) == len(self.dices_2) and len(
            self.dices_1) * 2 == STEPS_IN_FUTURE, "Programming Error!"

        next_state = current_state = self.start_state

        while current_state is not None and next_state is not None:

            for step in range(current_state.step, STEPS_IN_FUTURE):
                current_player = 2 if not (step % 2 == 0) else 1
                other_player = 2 if current_player == 1 else 1

                dice = self.dices[current_player][step // 2]

                self.simulate_step(current_player, other_player, current_state, dice, step)

                if step != STEPS_IN_FUTURE - 1:
                    # Get next child
                    # Only do this, if it is not the last step

                    next_state = self.state_list.get_next_child(current_state)

                    if next_state is not None:
                        current_state = next_state

            next_state = None
            while next_state is None and current_state is not None and current_state.child_iter <= len(
                    current_state.children):
                current_state = self.state_list.get_parent(current_state)
                if current_state is not None:
                    next_state = self.state_list.get_next_child(current_state)
            current_state = next_state

        if VISUALIZE:
            self.visualize()


if __name__ == "__main__":
    simulation = MinimaxSimulation()
    simulation.start()
