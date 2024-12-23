import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import graphviz
import numpy

# ----- User parameters ----- #

# Rules: https://www.mastersofgames.com/rules/royal-ur-rules.htm
# Rules from Tom Scott vs. Finkel
STEPS_IN_FUTURE = 5
PLAYER_1_MIN = True
ROSETTE_9_IS_SAFE = True

# Evaluation hyperparameter
EVAL_POINT_FINISH = 100
EVAL_POINT_START = -5
EVAL_MULTIPLIER_ROSETTE = 1.5
EVAL_MULTIPLIER_KILLABLE = 10
EVAL_MULTIPLIER_ATTACKER = -1.5
EVAL_ADDER_KILL_HAPPENS = 100

# Visualization
VISUALIZE = False
VIZ_THROWS = [2, 3]

# ----- Constants ----- #

PLACE_FINISH = numpy.uint8(0)
PLACE_START = numpy.uint8(1)
NUM_OF_PIECES_PER_PLAYER = 5
MASK_PIECE_0 = numpy.uint32(0xF)
MASK_PIECE_1 = numpy.uint32(MASK_PIECE_0 << 4)
MASK_PIECE_2 = numpy.uint32(MASK_PIECE_1 << 4)
MASK_PIECE_3 = numpy.uint32(MASK_PIECE_2 << 4)
MASK_PIECE_4 = numpy.uint32(MASK_PIECE_3 << 4)
ROSETTE_SAFE = 0x9
ROSETTES = [5, 15]


def _get_file_handler(filename: str) -> logging.FileHandler:
    handler = logging.FileHandler(filename, mode="w")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    return handler


logger_out = logging.getLogger("Minimax-Out")
logger_out.addHandler(_get_file_handler("out.txt"))
logger_out.setLevel(logging.INFO)

logger_eval = logging.getLogger("Minimax-Eval")
logger_eval.addHandler(_get_file_handler("out_eval.txt"))
logger_eval.setLevel(logging.INFO)


class ListIndexSafe(list):
    def index_safe(self, *args, **kwargs) -> int:
        try:
            return self.index(*args, **kwargs)
        except ValueError:
            return -1


def get_piece_mask(piece_index: int) -> numpy.uint32:
    if piece_index == 0:
        return MASK_PIECE_0

    if piece_index == 1:
        return MASK_PIECE_1

    if piece_index == 2:
        return MASK_PIECE_2

    if piece_index == 3:
        return MASK_PIECE_3

    if piece_index == 4:
        return MASK_PIECE_4


@dataclass
class State:
    score_1: int
    score_2: int
    pieces_1: numpy.uint32
    pieces_2: numpy.uint32
    current_player: int
    other_player: int
    dice: int = field(init=False, default=-1)
    moved_piece: int = field(init=False, default=-1)
    second_throw: bool = field(init=False, default=False)
    parent_pos: Optional[int] = field(init=False, default=None)
    pos: int = field(init=False, default=-1)
    children: List[int] = field(init=False, default_factory=list)
    child_iter: int = field(init=False, default=-1)
    eval: float = field(init=False, default=0)

    def copy(self) -> "State":
        score_1 = self.score_1
        score_2 = self.score_2
        pieces_1 = self.pieces_1.copy()
        pieces_2 = self.pieces_2.copy()
        current_player = self.current_player
        other_player = self.other_player

        state = State(score_1, score_2, pieces_1, pieces_2, current_player,
                      other_player)

        return state

    def __str__(self) -> str:
        return (f"\tPosition: {self.pos}\n"
                f"\tCurrent player: {self.current_player} - other_player: {self.other_player}\n"
                f"\tIs Second Throw: {self.second_throw}\n"
                f"\tPieces 1: {self.pieces_1} - Pieces 2: {self.pieces_2}\n"
                f"\tScore 1: {self.score_1} - Score 2: {self.score_2}\n")

    def check_win(self, player: int) -> None:
        score_player = player_based_list(self.score_1, self.score_2)[player]

        if score_player == NUM_OF_PIECES_PER_PLAYER:
            # TODO: Was hier?
            logger_out.info("Win - Keine Ahnung was jetzt")
            sys.exit(0)

    def swap_player(self) -> None:
        tmp = self.current_player
        self.current_player = self.other_player
        self.other_player = tmp

    def piece_move(self, player: int, piece_index: int, to_index: numpy.uint8) -> None:
        # set new piece place
        piece_mask_clear = ~get_piece_mask(piece_index)

        finished = to_index == PLACE_FINISH

        if player == 1:
            self.pieces_1 &= piece_mask_clear
            self.pieces_1 |= numpy.uint32(to_index) << (4 * piece_index)

            if finished:
                self.score_1 += 1

        elif player == 2:
            self.pieces_2 &= piece_mask_clear
            self.pieces_2 |= numpy.uint32(to_index) << (4 * piece_index)

            if finished:
                self.score_2 += 1
        else:
            raise Exception(f"Invalid player {player}")


def player_based_list(e1, e2) -> list:
    return [None, e1, e2]


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

    def get_next_child(self, state: State) -> Optional[State]:
        state.child_iter += 1
        try:
            return self.states[state.children[state.child_iter]]
        except IndexError:
            return None

    def get(self, index: int) -> State:
        return self.states[index]

    def __len__(self):
        return len(self.states)


def piece_cannot_finish(current_pos: int, dice: int) -> bool:
    return current_pos + dice > 16


def piece_can_finish(current_pos: int, dice: int) -> bool:
    return current_pos + dice == 16


def any_piece_on_field(pieces: numpy.uint32, pos: int) -> Tuple[bool, Optional[int]]:
    ret_flag = False
    piece_index = None

    if pieces & MASK_PIECE_0 == pos:
        piece_index = 0
        ret_flag = True
    elif (pieces & MASK_PIECE_1) >> (4 * 1) == pos:
        piece_index = 1
        ret_flag = True
    elif (pieces & MASK_PIECE_2) >> (4 * 2) == pos:
        piece_index = 2
        ret_flag = True
    elif (pieces & MASK_PIECE_3) >> (4 * 3) == pos:
        piece_index = 3
        ret_flag = True
    elif (pieces & MASK_PIECE_4) >> (4 * 4) == pos:
        piece_index = 4
        ret_flag = True

    return ret_flag, piece_index


def piece_cannot_move_current_player_is_on_field(current_state: State, current_pos: int) -> bool:
    pieces_current_player = player_based_list(current_state.pieces_1, current_state.pieces_2)[
        current_state.current_player]

    next_pos = current_pos + current_state.dice

    return any_piece_on_field(pieces_current_player, next_pos)[0]


def piece_cannot_move_other_player_is_on_safe(current_state: State, current_pos: int) -> bool:
    if current_pos + current_state.dice != ROSETTE_SAFE:
        # Player will not move to Rosette Safe Space
        return False

    return other_player_is_on_field(current_state, current_pos)[0]


def other_player_is_on_field(current_state: State, current_pos: int) -> Tuple[bool, Optional[int]]:
    next_pos = current_pos + current_state.dice

    if not (6 <= next_pos <= 13):
        # War zone is only between 6 and 13
        return False, None

    pieces_other = player_based_list(current_state.pieces_1, current_state.pieces_2)[current_state.other_player]

    return any_piece_on_field(pieces_other, next_pos)


class MinimaxSimulation:
    # evaluation
    base_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, EVAL_POINT_FINISH,
                   EVAL_POINT_START]
    rosette_bonus = [value * EVAL_MULTIPLIER_ROSETTE for value in
                     [1 / 16, 1 / 4, 3 / 8, 1, 1 / 16, 1 / 4, 3 / 8, 1, 0, 1 / 16, 1 / 4, 3 / 8, 1,
                      0, 0]]
    kill_distances_multiplier = [None, 1 / 4, 3 / 8, 1 / 4, 1 / 16]

    def __init__(self) -> None:
        self.state_list = StateList()

        # ----- Start State ----- #

        # Indices of game_board places for both players
        pieces_1 = numpy.uint32(PLACE_START) | numpy.uint32(PLACE_START) << (4 * 1) | numpy.uint32(PLACE_START) << (
                4 * 2) | numpy.uint32(PLACE_START) << (4 * 3) | numpy.uint32(PLACE_START) << (4 * 4)
        pieces_2 = numpy.uint32(PLACE_START) | numpy.uint32(PLACE_START) << (4 * 1) | numpy.uint32(PLACE_START) << (
                4 * 2) | numpy.uint32(PLACE_START) << (4 * 3) | numpy.uint32(PLACE_START) << (4 * 4)

        # Number of pieces in finish for both players
        score_1 = 0
        score_2 = 0

        self.start_state = self.state_list.add_new_state(
            State(score_1, score_2, pieces_1, pieces_2, 1, 2))

    def evaluation(self, state_source: State, state_new: State) -> float:
        # Simulation will swap the player if no second throw
        # The evaluation should use the original "current_player" and "other_player"
        if state_new.second_throw:
            current_player = state_new.current_player
            other_player = state_new.other_player
        else:
            current_player = state_new.other_player
            other_player = state_new.current_player

        places = player_based_list(state_new.pieces_1, state_new.pieces_2)
        paths = player_based_list(self.path_1, self.path_2)

        places_current_player = places[current_player]
        places_other_player = places[other_player]

        paths_current_player = paths[current_player]
        paths_other_player = paths[other_player]

        points_total = 0
        for piece_place_current in places_current_player:

            if piece_place_current in [PLACE_START, PLACE_FINISH]:
                path_index_current = piece_place_current
            else:
                path_index_current = paths_current_player.index(piece_place_current)

            # self.path_points + (self.rosette_multiplier * POINT_ROSETTE_MULTIPLIER)
            # ==
            # self.path_points + self.rosette_bonus
            points_total += MinimaxSimulation.base_points[path_index_current] + \
                            MinimaxSimulation.rosette_bonus[path_index_current]

            if path_index_current in [PLACE_START, PLACE_FINISH]:
                continue

            # Killable

            path_kill_range = paths_current_player[path_index_current:path_index_current + 4]
            killable_pieces_of_other_player = [piece_place_other for piece_place_other in
                                               places_other_player if
                                               piece_place_other in path_kill_range]
            count_killable = len(killable_pieces_of_other_player)
            points_total += count_killable * EVAL_MULTIPLIER_KILLABLE

            # Attackers

            if 6 <= piece_place_current <= 13:
                # Beispiel: piece_place_current == 8
                # path_attacker_range == [15, 14, 6, 7]
                path_attacker_range = paths_other_player[path_index_current - 4:path_index_current]
                attacker_pieces_of_other_player = [piece_place_other for piece_place_other in
                                                   places_other_player if
                                                   piece_place_other in path_attacker_range]
                count_attacker = len(attacker_pieces_of_other_player)
                points_total += count_attacker * EVAL_MULTIPLIER_ATTACKER

        # ------------ Improvements of state ------------ #

        other_pieces_source = player_based_list(state_source.pieces_1, state_source.pieces_2)[other_player]
        other_pieces_new = player_based_list(state_new.pieces_1, state_new.pieces_2)[other_player]

        count_other_pieces_source_start = sum([1 for a in other_pieces_source if a == PLACE_START])
        count_other_pieces_new_start = sum([1 for a in other_pieces_new if a == PLACE_START])

        kill_happens = count_other_pieces_new_start != count_other_pieces_source_start

        points_total += kill_happens * EVAL_ADDER_KILL_HAPPENS

        return points_total

    @staticmethod
    def simulate_step(current_state: State, piece_index: int, dice: int) -> Optional[State]:
        # current_state is a copy of the current state and can therefore be modified

        current_state.dice = dice
        current_state.moved_piece = piece_index

        # Reset second throw
        current_state.second_throw = False

        current_player = current_state.current_player
        other_player = current_state.other_player

        pieces_current_player = player_based_list(current_state.pieces_1, current_state.pieces_2)[
            current_player]

        place_current_piece = (pieces_current_player & get_piece_mask(piece_index)) >> (4 *piece_index)

        if dice == 0:
            # No movement

            state_new = current_state

        elif place_current_piece == PLACE_FINISH:
            # Piece is already in finish
            # -> This is not a valid move

            return None

        elif piece_cannot_finish(place_current_piece, dice):
            # Move cannot be done, because the piece has to be finished perfectly
            # -> This is not a valid move

            return None

        elif piece_can_finish(place_current_piece, dice):
            # Piece is in finish with next move

            # Piece moves to finish
            current_state.piece_move(current_player, piece_index, PLACE_FINISH)

            state_new = current_state

        elif piece_cannot_move_current_player_is_on_field(current_state, place_current_piece):
            # Move cannot be done, on the field is already a piece of the current_player,
            # -> This is not a valid move

            return None

        elif ROSETTE_9_IS_SAFE and piece_cannot_move_other_player_is_on_safe(current_state,
                                                                             place_current_piece):
            # Move cannot be done, because this rosette is a safe spot for the other player
            # -> This is not a valid move

            return None

        else:
            # Valid move and not in finish

            flag, piece_index_other_player = other_player_is_on_field(current_state, place_current_piece)
            if flag:
                # Other player will be caught and returned to start
                current_state.piece_move(other_player, piece_index_other_player, PLACE_START)

            if place_current_piece in ROSETTES:
                current_state.second_throw = True

            # current player moves from current place to new place
            current_state.piece_move(current_player, piece_index, place_current_piece + dice)

            state_new = current_state

        if not current_state.second_throw:
            state_new.swap_player()

        return state_new

    def visualize(self) -> None:
        graph = graphviz.Graph(name="Graph")

        for state in self.state_list:
            if state.pos == 0:
                current_player = 1
            elif state.second_throw:
                current_player = state.current_player
            else:
                current_player = state.other_player

            color = "green" if current_player == 1 else "red"
            graph.node(str(state.pos), f"ID: {state.pos}\nS: {state.eval}\nD: {state.dice}\nMP: {state.moved_piece}",
                       _attributes={"color": color})
        for state in self.state_list:
            for child in state.children:
                graph.edge(str(state.pos), str(child))

        graph.view()

    def visualize_path(self) -> None:
        graph = graphviz.Graph(name="Graph_Path")

        root_node = self.state_list.get(0)

        nodes_to_visualize = [root_node]

        current_nodes = [root_node]
        for throw in VIZ_THROWS:
            new_nodes = []
            for current_node in current_nodes:
                children = [self.state_list.get(child) for child in current_node.children]
                children_with_throw = [child for child in children if child.dice == throw]
                nodes_to_visualize.extend(children_with_throw)
                new_nodes.extend(children_with_throw)

            current_nodes = new_nodes

        for node in nodes_to_visualize:
            if node.pos == 0:
                current_player = 1
            elif node.second_throw:
                current_player = node.current_player
            else:
                current_player = node.other_player

            color = "green" if current_player == 1 else "red"
            graph.node(str(node.pos), f"ID: {node.pos}\nS: {node.eval}\nD: {node.dice}\nMP: {node.moved_piece}",
                       _attributes={"color": color})

        for state in nodes_to_visualize:
            for child in state.children:
                if child in [node.pos for node in nodes_to_visualize]:
                    graph.edge(str(state.pos), str(child))
        graph.view()

    def start(self) -> None:
        current_state = self.start_state

        current_step = 0

        while current_state is not None:
            for step in range(current_step, STEPS_IN_FUTURE):
                logger_out.info(f"Step: {step}")
                logger_out.info(f"Current state: \n{current_state}")

                # ----- For each piece of current player ----- #

                for piece_index in range(0, NUM_OF_PIECES_PER_PLAYER):
                    logger_out.info(f"Simulate piece '{piece_index}'")

                    # ----- For each possible dice throw ----- #

                    for dice in range(0, 4 + 1):
                        logger_out.info(f"Simulate dice '{dice}'")

                        state_new = self.simulate_step(current_state.copy(), piece_index, dice)

                        if state_new is None:
                            logger_out.info("Movement not possible")
                            continue

                        state_new.parent_pos = current_state.pos
                        state_new = self.state_list.add_new_state(state_new)
                        current_state.children.append(state_new.pos)

                        # ----- Check win ----- #
                        state_new.check_win(current_state.current_player)

                        # ----- Evaluation ----- #
                        # score = self.evaluation(current_state, state_new)
                        # logger_eval.info(f"{step},{score}")
                        # state_new.eval = score

                        logger_out.info(f"Simulated state: \n{state_new}")

                    # ----- "Normalize" all evaluation scores of this piece ----- #
                    # if current_state.current_player == 1 and PLAYER_1_MIN:
                    #    normalized_eval = min([self.state_list.get(index).eval for index in current_state.children])
                    # else:
                    #    normalized_eval = max([self.state_list.get(index).eval for index in current_state.children])
                    # for child in [self.state_list.get(index) for index in current_state.children]:
                    #   child.eval = normalized_eval

                if len(current_state.children) == 0:
                    # No piece can move. This leads to no child, which would be incorrect. Therefore the same state will be the next state, but with player swap
                    # TODO: what should evalulation score be?
                    state_new = current_state.copy()
                    state_new.swap_player()

                    state_new.parent_pos = current_state.pos
                    state_new = self.state_list.add_new_state(state_new)
                    current_state.children.append(state_new.pos)

                if step != STEPS_IN_FUTURE - 1:
                    # Get next child
                    # Only do this, if it is not the last step

                    next_state = self.state_list.get_next_child(current_state)

                    # TODO: Korrekt?
                    assert next_state is not None

                    current_state = next_state

            current_step = STEPS_IN_FUTURE

            next_state = None
            while next_state is None and current_state is not None and current_state.child_iter <= len(
                    current_state.children):
                current_step -= 1
                current_state = self.state_list.get_parent(current_state)
                if current_state is not None:
                    next_state = self.state_list.get_next_child(current_state)
            current_state = next_state

        if VISUALIZE:
            self.visualize()
            self.visualize_path()

        print(len(self.state_list))


if __name__ == "__main__":
    import datetime
    print(datetime.datetime.now())
    simulation = MinimaxSimulation()
    simulation.start()
    print(datetime.datetime.now())
