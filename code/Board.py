#!/usr/bin/python

import numpy as np

class State:
    Empty = 0
    Occupied = 1

class Color:
    Black = 0
    White = 1


color_names = {Color.Black:"Black", Color.White:"White", }

flipped_color = {Color.Black: Color.White, Color.White: Color.Black }

class IllegalMoveException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class Board:
    def __init__(self, N):
        self.N = N
        self.num_dirs = 4
        self.empty_edges = N * (N + 1) * 2
        self.dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.clear()

    def clear(self):
        self.edges = np.empty((self.N, self.N, self.num_dirs), dtype=np.int8)
        self.edges.fill(State.Empty)
        self.move_list = []
        self.score = {Color.Black: 0, Color.White: 0}
        self.color_to_play = Color.Black

    def __getitem__(self, index):
        return self.edges[index]

    def is_on_board(self, x, y, dir):
        return 0 <= x and x < self.N and 0 <= y and y < self.N and 0 <= dir and dir < self.num_dirs

    def find_share_edge(self, x, y, dir):
        none = (None, None, None)
        if (x == 0 and dir == 0): return none
        if (y == 0 and dir == 3): return none
        if (x == self.N - 1 and dir == 2): return none
        if (y == self.N - 1 and dir == 1): return none

        if (dir == 0): return (x - 1, y, 2)
        if (dir == 1): return (x, y + 1, 3)
        if (dir == 2): return (x + 1, y, 0)
        if (dir == 3): return (x, y - 1, 1)
        return none

    def count_fenced(self, x, y):
        sum = 0
        for d in range(0, self.num_dirs):
            sum = sum + self.edges[x, y, d]
        return sum

    def check_game_end(self):
        return (self.empty_edges == 0) or (self.score[Color.Black] * 2 > self.N * self.N) or ((self.score[Color.White] * 2 > self.N * self.N))

    def try_make_move(self, x, y, dir, color, actually_execute):
        assert color == Color.White or color == Color.Black

        if (not self.is_on_board(x, y, dir)): return False
        if (self.edges[x, y, dir] != State.Empty): return False

        (nx, ny, nd) = self.find_share_edge(x, y, dir)


        if (actually_execute):
            self.move_list.append([x, y, dir])
            self.edges[x, y, dir] = State.Occupied
            self.empty_edges = self.empty_edges - 1

            count = self.count_fenced(x, y)
            flip = True
            if (count == self.num_dirs):
                flip = False
                self.score[color] += 1

            if (nx != None):
                self.edges[nx, ny, nd] = State.Occupied
                count = self.count_fenced(nx, ny)
                if (count == self.num_dirs):
                    self.score[color] += 1
                    flip = False

            if (flip == True):
                self.color_to_play = flipped_color[color]
        return True

    def make_move(self, x, y, dir, color):
        if not self.try_make_move(x, y, dir, color, actually_execute=True):
            raise IllegalMoveException("%s player (%d, %d, dir=%d is illegal" % (color_names[color], x, y, dir))

    def play_is_legal(self, x, y, dir, color):
        return self.try_make_move(x, y, dir, color, actually_execute=False)


    def show(self):
        state_strings = {
            (0, State.Occupied): '\033[31m--\033[0m',
            (0, State.Empty): '\033[37m--\033[0m',
            (1, State.Occupied): '\033[31m|\033[0m',
            (1, State.Empty): '\033[37m|\033[0m'
        }

        # print board
        print("Score: ", self.score[Color.Black], self.score[Color.White], color_names[self.color_to_play], "to play")
        for x in range(self.N + 1):
            if (x < self.N):
                for y in range(self.N):
                    print("+"+state_strings[0, self.edges[x, y, 0]], end="")
                print("+")
                for y in range(self.N):
                    print(state_strings[1, self.edges[x, y, 3]]+"  ", end="")
                print(state_strings[1, self.edges[x, self.N - 1, 1]])
            else:
                for y in range(self.N):
                    print("+"+state_strings[0, self.edges[x - 1, y, 2]], end="")
                print("+")




def show_sequence(board, moves, first_color):
    board.clear()
    color = first_color
    for x,y,dir in moves:
        legal = board.make_move(x, y, dir, color)
        board.show()
        color = board.color_to_play


def test_Board():
    board = Board(5)
    show_sequence(board, [(0, 0, 1), (0, 0, 0), (0, 0, 2), (0, 0, 3), (2, 2, 0),(2, 2, 1),(2, 2, 2),(2, 2, 3)], Color.Black)



if __name__ == "__main__":
    test_Board()



