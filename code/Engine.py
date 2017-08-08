from Board import Board
import copy
import random

class BaseEngine(object):
    def __init__(self):
        self.board = None
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
    def set_board_size(self, N):
        self.board = Board(N)
        return True

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
    def __init__(self):
        super(IdiotEngine,self).__init__()

    def name(self):
        return "IdiotEngine"

    def version(self):
        return "1.0"

    def pick_move(self, color):
        moves = []
        for x in range(self.board.N):
            for y in range(self.board.N):
                for d in range(self.board.num_dirs):
                    if self.board.play_is_legal(x, y, d, color):
                        moves.append([x, y, d, color])
        return random.choice(moves)


class StrongerEngine(BaseEngine):
    def __init__(self):
        super(StrongerEngine,self).__init__()

    def name(self):
        return "StrongerEngine"

    def version(self):
        return "1.0"

    def pick_move(self, color):
        point_moves = []
        def_moves = []
        other_moves = []
        for x in range(self.board.N):
            for y in range(self.board.N):
                for d in range(self.board.num_dirs):
                    if self.board.play_is_legal(x, y, d, color):
                        (nx, ny ,nd) = self.board.find_share_edge(x, y, d)
                        points = False
                        defs = True
                        if (self.board.count_fenced(x, y) == self.board.num_dirs - 1):
                            points = True
                        if (self.board.count_fenced(x, y) == self.board.num_dirs - 2):
                            defs = False
                        if (nx != None and self.board.count_fenced(nx, ny) == self.board.num_dirs - 1):
                            points = True
                        if (nx != None and self.board.count_fenced(nx, ny) == self.board.num_dirs - 2):
                            defs = False

                        if (points == True):
                            point_moves.append([x, y, d, color])
                        elif (defs == True):
                            def_moves.append([x, y, d, color])
                        else:
                            other_moves.append([x, y, d, color])

        if point_moves:
            return random.choice(point_moves)
        elif def_moves:
            return random.choice(def_moves)
        else:
            return random.choice(other_moves)


