import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import graphviz

output_file = Path() / "output.txt"
output_file.open("w").write("")
output_file_eval = Path() / "output_eval.txt"
output_file_eval.open("w").write("")


def print_out(text="") -> None:
    text = str(text)
    text += "\n"
    output_file.open("a").write(text)


def print_eval(text="") -> None:
    text = str(text)
    text += "\n"
    output_file_eval.open("a").write(text)


class ListIndexSafe(list):
    def index_safe(self, *args, **kwargs) -> int:
        try:
            return self.index(*args, **kwargs)
        except ValueError:
            return -1


# Rules: https://www.mastersofgames.com/rules/royal-ur-rules.htm
# Rules from Tom Scott vs. Finkel
NUM_OF_PIECES_PER_PLAYER = 5
STEPS_IN_FUTURE = 2
PLAYER_1_MIN = True
VISUALIZE = True
ROSETTE_9_IS_SAFE = True
START_STEP = 0

# Hyperparameter Bewertung
EVAL_POINT_FINISH = 100
EVAL_POINT_START = -5
EVAL_MULTIPLIER_ROSETTE = 1.5
EVAL_MULTIPLIER_KILLABLE = 10
EVAL_MULTIPLIER_ATTACKER = -1.5
EVAL_ADDER_KILL_HAPPENS = 100

# Constants
PLACE_ROSETTE = 3
PLACE_ROSETTE_SAFE = 6 if ROSETTE_9_IS_SAFE else PLACE_ROSETTE
PLACE_START = -1
PLACE_FINISH = -2

# Visualization
VIZ_THROWS = [2,3]


@dataclass
class State:
    game_board: List[int]
    score_1: int
    score_2: int
    pieces_1: List[int]
    pieces_2: List[int]
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
        game_board = self.game_board.copy()
        score_1 = self.score_1
        score_2 = self.score_2
        pieces_1 = self.pieces_1.copy()
        pieces_2 = self.pieces_2.copy()
        current_player = self.current_player
        other_player = self.other_player

        state = State(game_board, score_1, score_2, pieces_1, pieces_2, current_player,
                      other_player)

        return state

    def __str__(self) -> str:
        return (f"\tPosition: {self.pos}\n"
                f"\tCurrent player: {self.current_player} - other_player: {self.other_player}\n"
                f"\tIs Second Throw: {self.second_throw}\n"
                f"\tBoard state: {self.game_board}\n"
                f"\tPieces 1: {self.pieces_1} - Pieces 2: {self.pieces_2}\n"
                f"\tScore 1: {self.score_1} - Score 2: {self.score_2}\n")

    def check_win(self, player: int) -> None:
        score_player = player_based_list(self.score_1, self.score_2)[player]

        if score_player == NUM_OF_PIECES_PER_PLAYER:
            # TODO: Was hier?
            print_out("Win - Keine Ahnung was jetzt")
            print("Win")
            sys.exit(0)

    def swap_player(self) -> None:
        tmp = self.current_player
        self.current_player = self.other_player
        self.other_player = tmp

    def piece_move(self, player: int, piece_index: int, from_index: int, to_index: int) -> None:
        if from_index != PLACE_START:
            self.game_board[from_index] -= player

        if to_index not in [PLACE_START, PLACE_FINISH]:
            self.game_board[to_index] += player

        # set new piece place
        pieces_pbl = player_based_list(self.pieces_1, self.pieces_2)
        pieces_pbl[player][piece_index] = to_index
        self.pieces_1 = pieces_pbl[1]
        self.pieces_2 = pieces_pbl[2]


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


class MinimaxSimulation:
    # evaluation
    base_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, EVAL_POINT_FINISH,
                   EVAL_POINT_START]
    rosette_bonus = [value * EVAL_MULTIPLIER_ROSETTE for value in
                     [1 / 16, 1 / 4, 3 / 8, 1, 1 / 16, 1 / 4, 3 / 8, 1, 0, 1 / 16, 1 / 4, 3 / 8, 1,
                      0, 0]]
    kill_distances_multiplier = [None, 1 / 4, 3 / 8, 1 / 4, 1 / 16]

    def __init__(self) -> None:
        # States:
        # 0 -> No player on this field
        # 1 -> Player1 on this field
        # 2 -> Player2 on this field
        # 3 -> Rosette (another throw)
        # 4 = 3 + 1 -> Player 1 on rosette
        # 5 = 3 + 2 -> Player 2 on rosette
        #self.game_board = [PLACE_ROSETTE, 0, 0, 0, PLACE_ROSETTE, 0, 1, 0, 2, PLACE_ROSETTE_SAFE, 1, 0, 0, 0,
        #                   PLACE_ROSETTE, 0, 0, 0, PLACE_ROSETTE, 0]
        self.game_board = [PLACE_ROSETTE, 0, 0, 0, PLACE_ROSETTE, 0, 0, 0, 0, PLACE_ROSETTE_SAFE, 1, 0, 0, 0,
                           PLACE_ROSETTE, 0, 0, 0, PLACE_ROSETTE, 0]

        # Indices of game_board path for both players
        self.path_1 = ListIndexSafe([3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 5, 4])
        self.path_2 = ListIndexSafe([17, 16, 15, 14, 6, 7, 8, 9, 10, 11, 12, 13, 19, 18])
        self.paths: List[ListIndexSafe] = player_based_list(self.path_1, self.path_2)

        self.state_list = StateList()

        # ----- Start State ----- #

        # Indices of game_board places for both players
        # -1 -> Start
        # -2 -> Finish
        pieces_1 = [PLACE_START] * NUM_OF_PIECES_PER_PLAYER
        #pieces_1[0] = 6
        #pieces_1[1] = 10

        pieces_2 = [PLACE_START] * NUM_OF_PIECES_PER_PLAYER
        #pieces_2[1] = 8

        # Number of pieces in finish for both players
        score_1 = 0
        score_2 = 0
        # Start state
        game_board = self.game_board

        self.start_state = self.state_list.add_new_state(
            State(game_board, score_1, score_2, pieces_1, pieces_2, 1, 2))

    def piece_cannot_finish(self, current_player: int, next_path_index: int) -> bool:
        path = self.paths[current_player]
        return next_path_index > len(path)

    def piece_can_finish(self, current_player: int, next_path_index: int) -> bool:
        path = self.paths[current_player]
        return next_path_index == len(path)

    def piece_cannot_move_current_player_is_on_field(self, current_state: State,
                                                     next_path_index: int) -> bool:
        next_place_index = self.paths[current_state.current_player][next_path_index]

        places_current_player = player_based_list(current_state.pieces_1, current_state.pieces_2)[
            current_state.current_player]

        return any([place == next_place_index for place in places_current_player])

    def get_next_path_index(self, current_player: int, place_index: int, dice: int) -> int:
        path = self.paths[current_player]
        return path.index_safe(place_index) + dice

    def piece_cannot_move_other_player_is_on_safe(self, current_state: State,
                                                  next_path_index: int) -> bool:
        path = self.paths[current_state.current_player]

        return current_state.game_board[
            path[next_path_index]] - current_state.other_player == PLACE_ROSETTE_SAFE

    def other_player_is_on_field(self, current_state: State, next_path_index: int) -> bool:
        next_place_index = self.paths[current_state.current_player][next_path_index]

        places_other_player = player_based_list(current_state.pieces_1, current_state.pieces_2)[
            current_state.other_player]

        return any([place == next_place_index for place in places_other_player])

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

    def simulate_step(self, current_state: State, piece_index: int, dice: int) -> Optional[State]:
        # current_state is a copy of the current state and can therefore be modified

        current_state.dice = dice
        current_state.moved_piece = piece_index

        # Reset second throw
        current_state.second_throw = False

        current_player = current_state.current_player
        other_player = current_state.other_player

        scores = player_based_list(current_state.score_1, current_state.score_2)
        pieces_current_player = player_based_list(current_state.pieces_1, current_state.pieces_2)[
            current_player]

        place_current_piece = pieces_current_player[piece_index]

        next_path_index = self.get_next_path_index(current_player, place_current_piece, dice)

        if dice == 0:
            # No movement

            state_new = current_state

        elif place_current_piece == PLACE_FINISH:
            # Piece is already in finish
            # -> This is not a valid move

            return None

        elif self.piece_cannot_finish(current_player, next_path_index):
            # Move cannot be done, because the piece has to be finished perfectly
            # -> This is not a valid move

            return None

        elif self.piece_can_finish(current_player, next_path_index):
            # Piece is in finish with next move

            # Piece moves to finish
            current_state.piece_move(current_player, piece_index, place_current_piece, PLACE_FINISH)
            scores[current_player] += 1

            state_new = current_state

        elif self.piece_cannot_move_current_player_is_on_field(current_state, next_path_index):
            # Move cannot be done, on the field is already a piece of the current_player,
            # -> This is not a valid move

            return None

        elif ROSETTE_9_IS_SAFE and self.piece_cannot_move_other_player_is_on_safe(current_state,
                                                                                  next_path_index):
            # Move cannot be done, because this rosette is a safe spot for the other player
            # -> This is not a valid move

            return None

        else:
            # Valid move and not in finish
            next_place_index = self.paths[current_state.current_player][next_path_index]

            if self.other_player_is_on_field(current_state, next_path_index):
                # Other player will be caught and returned to start

                piece_index_other_player = \
                    player_based_list(current_state.pieces_1, current_state.pieces_2)[
                        other_player].index(next_place_index)
                current_state.piece_move(other_player, piece_index_other_player, next_place_index,
                                         PLACE_START)

            if current_state.game_board[next_place_index] == PLACE_ROSETTE:
                current_state.second_throw = True

            # current player moves from current place to new place
            current_state.piece_move(current_player, piece_index, place_current_piece,
                                     next_place_index)

            state_new = current_state

        if not current_state.second_throw:
            state_new.swap_player()

        return state_new

    def visualize(self) -> None:
        graph = graphviz.Graph()

        for state in self.state_list:
            if state.pos == 0:
                current_player = 1
            elif state.second_throw:
                current_player = state.current_player
            else:
                current_player = state.other_player

            color = "green" if current_player == 1 else "red"
            graph.node(str(state.pos), f"{state.eval}\n{state.dice}\n{state.moved_piece}", _attributes={"color": color})

        for state in self.state_list:
            for child in state.children:
                graph.edge(str(state.pos), str(child))

        graph.view()

    def visualize_path(self) -> None:
        graph = graphviz.Graph(name="Graph2")

        visualized_children = []

        root = self.state_list.get(0)
        visualized_children.append(root)
        childs_step_1 = [self.state_list.get(child_pos) for child_pos in root.children  if self.state_list.get(child_pos).dice == VIZ_THROWS[0]]
        visualized_children.extend(childs_step_1)
        for child_1 in childs_step_1:
            visualized_children.extend([self.state_list.get(child_pos) for child_pos in child_1.children  if self.state_list.get(child_pos).dice == VIZ_THROWS[1]])

        for state in visualized_children:
            if state.pos == 0:
                current_player = 1
            elif state.second_throw:
                current_player = state.current_player
            else:
                current_player = state.other_player

            color = "green" if current_player == 1 else "red"
            graph.node(str(state.pos), f"{state.pos}\n{state.eval}\n{state.dice}\n{state.moved_piece}", _attributes={"color": color})

        for state in visualized_children:
            for child in state.children:
                graph.edge(str(state.pos), str(child))
        graph.view()

    def start(self) -> None:
        current_state = self.start_state

        current_step = START_STEP

        while current_state is not None:
            for step in range(current_step, STEPS_IN_FUTURE):
                print_out(f"Step: {step}")
                print_out(f"Current state: \n{current_state}")

                # ----- For each piece of current player ----- #

                for piece_index in range(0, NUM_OF_PIECES_PER_PLAYER):
                    print_out(f"Simulate piece '{piece_index}'")

                    # ----- For each possible dice throw ----- #

                    for dice in range(0, 4 + 1):
                        print_out(f"Simulate dice '{dice}'")

                        state_new = self.simulate_step(current_state.copy(), piece_index, dice)

                        if state_new is None:
                            print_out("Movement not possible")
                            continue

                        state_new.parent_pos = current_state.pos
                        state_new = self.state_list.add_new_state(state_new)
                        current_state.children.append(state_new.pos)

                        # ----- Check win ----- #
                        state_new.check_win(current_state.current_player)

                        # ----- Evaluation ----- #
                        score = self.evaluation(current_state, state_new)
                        print_eval(f"{step},{score}")
                        state_new.eval = score

                        print_out(f"Simulated state: \n{state_new}")

                    # ----- "Normalize" all evaluation scores of this piece ----- #
                    #if current_state.current_player == 1 and PLAYER_1_MIN:
                    #    normalized_eval = min([self.state_list.get(index).eval for index in current_state.children])
                    #else:
                    #    normalized_eval = max([self.state_list.get(index).eval for index in current_state.children])
                    #for child in [self.state_list.get(index) for index in current_state.children]:
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


if __name__ == "__main__":
    simulation = MinimaxSimulation()
    simulation.start()
