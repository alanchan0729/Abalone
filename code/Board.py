#!/usr/bin/python

import numpy as np

class State:
    Empty = 0
    Occupied = 1

class Color:
    Black = 0
    White = 1

class Direction:
    Horizontal = 0
    Vertical = 1


color_names = {Color.Black:"Black", Color.White:"White", }

flipped_color = {Color.Black: Color.White, Color.White: Color.Black }

class IllegalMoveException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class Board:
    def __init__(self, N):
        self.N = N
        self.num_dirs = 4
        self.move_dirs = 2
        self.clear()

    def clear(self):

        self.edges = np.empty((self.N + 1, self.N + 1, 2), dtype=np.int8)
        self.edges.fill(State.Empty)
        self.empty_edges = self.N * (self.N + 1) * 2
        self.move_list = []
        self.score = {Color.Black: 0, Color.White: 0}
        self.color_to_play = Color.Black

    def __getitem__(self, index):
        return self.edges[index]

    def is_on_board(self, x, y, dir):
        if (dir == None):
            return (0 <= x) and (x < self.N) and (0 <= y) and (y < self.N)
        if (dir == Direction.Vertical):
            return (0 <= x) and (x < self.N) and (0 <= y) and (y < self.N + 1)
        elif (dir == Direction.Horizontal):
            return (0 <= x) and (x < self.N + 1) and (0 <= y) and (y < self.N)
        else:
            return False


    def count_fenced(self, x, y):
        return self.edges[x, y, Direction.Horizontal] + self.edges[x + 1, y, Direction.Horizontal]\
               + self.edges[x, y, Direction.Vertical] + self.edges[x, y + 1, Direction.Vertical]

    def find_share_grid(self, x, y, dir):
        if (dir == Direction.Horizontal):
            return x - 1, y
        elif (dir == Direction.Vertical):
            return x, y - 1
        else:
            return None, None

    def check_game_end(self):
        return (self.empty_edges == 0) or (self.score[Color.Black] * 2 > self.N * self.N) or ((self.score[Color.White] * 2 > self.N * self.N))

    def try_make_move(self, x, y, dir, color, actually_execute):
        assert color == Color.White or color == Color.Black

        if (not self.is_on_board(x, y, dir)): return False
        if (self.edges[x, y, dir] != State.Empty): return False


        if (actually_execute):
            self.move_list.append([x, y, dir])
            self.edges[x, y, dir] = State.Occupied
            self.empty_edges = self.empty_edges - 1

            flip = True

            if (self.is_on_board(x, y, None)):
                count = self.count_fenced(x, y)
                if (count == self.num_dirs):
                    flip = False
                    self.score[color] += 1

            nx, ny = self.find_share_grid(x, y, dir)


            if (self.is_on_board(nx, ny, None)):
                count = self.count_fenced(nx, ny)
                if (count == self.num_dirs):
                    flip = False
                    self.score[color] += 1

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
                    print("+"+state_strings[0, self.edges[x, y, Direction.Horizontal]], end="")
                print("+")
                for y in range(self.N):
                    print(state_strings[1, self.edges[x, y, Direction.Vertical]]+"  ", end="")
                print(state_strings[1, self.edges[x, self.N, Direction.Vertical]])
            else:
                for y in range(self.N):
                    print("+"+state_strings[0, self.edges[x, y, Direction.Horizontal]], end="")
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
    show_sequence(board, [(0, 0, 1), (0, 0, 0), (0, 1, 1), (1, 0, 0),
                          (2, 2, 0),(2, 2, 1),(3, 2, 0),(2, 3, 0),
                          (2, 4, 1), (3, 3, 0), (2, 3, 1)], Color.Black)



if __name__ == "__main__":
    test_Board()



