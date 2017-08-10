from Board import Board
import copy
import random

class BaseEngine(object):
    def __init__(self, N):
        self.board = Board(N)
        self.state_stack = []

    def push_state(self):
        self.state_stack.append(copy.deepcopy(self.board))

    # subclasses must override this
    def name(self):
        assert False

    # subclasses must override this
    def version(self):
        assert False

    # subclasses may override to only accept
    # certain board sizes. They should call this
    # base method.

    def clear_board(self):
        self.board.clear()
        self.state_stack = []


    def move_played(self, x, y, dir, color):
        self.push_state()
        self.board.make_move(x, y, dir, color)
        # self.board.show()

    # subclasses must override this
    def pick_move(self, color):
        assert False

    def quit(self):
        pass

    def supports_final_status_list(self):
        return False


class IdiotEngine(BaseEngine):
    def __init__(self, N):
        super(IdiotEngine,self).__init__(N)

    def name(self):
        return "IdiotEngine"

    def version(self):
        return "1.0"

    def pick_move(self, color, show=False, pick_best=False):
        moves = []
        for x in range(self.board.N + 1):
            for y in range(self.board.N + 1):
                for d in range(self.board.move_dirs):
                    if self.board.play_is_legal(x, y, d, color):
                        moves.append([x, y, d, color])
        return random.choice(moves)


class StrongerEngine(BaseEngine):
    def __init__(self, N, epsilon):
        super(StrongerEngine,self).__init__(N)
        self.epsilon = epsilon

    def name(self):
        return "StrongerEngine (%.2f)" % self.epsilon

    def version(self):
        return "1.0"

    def pick_move(self, color, show=False, pick_best=False):
        point_moves = []
        def_moves = []
        other_moves = []
        all_moves = []
        for x in range(self.board.N + 1):
            for y in range(self.board.N + 1):
                for d in range(self.board.move_dirs):
                    if self.board.play_is_legal(x, y, d, color):
                        all_moves.append([x, y, d, color])
                        nx, ny = self.board.find_share_grid(x, y, d)
                        points = False
                        defs = True

                        if (self.board.is_on_board(x, y, None)):
                            if (self.board.count_fenced(x, y) == self.board.num_dirs - 1):
                                points = True
                            if (self.board.count_fenced(x, y) == self.board.num_dirs - 2):
                                defs = False

                        if (self.board.is_on_board(nx, ny, None)):
                            if (self.board.count_fenced(nx, ny) == self.board.num_dirs - 1):
                                points = True
                            if (self.board.count_fenced(nx, ny) == self.board.num_dirs - 2):
                                defs = False

                        if (points == True):
                            point_moves.append([x, y, d, color])
                        elif (defs == True):
                            def_moves.append([x, y, d, color])
                        else:
                            other_moves.append([x, y, d, color])

        r = random.random()
        if (r < self.epsilon):
            return random.choice(all_moves)
        else:
            if point_moves:
                return random.choice(point_moves)
            elif def_moves:
                return random.choice(def_moves)
            else:
                return random.choice(other_moves)


