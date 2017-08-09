#!/usr/bin/python

import numpy as np

class Color:
    Empty = 0
    Black = 1
    White = 2

color_names = { Color.Empty:"Empty", Color.Black:"Black", Color.White:"White", }

flipped_color = { Color.Empty: Color.Empty, Color.Black: Color.White, Color.White: Color.Black }

num_dirs = 6
dirs = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]



class IllegalMoveException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class Board:
    def __init__(self, N, opening):
        self.N = N
        self.opening = opening
        self.clear()

    def clear(self):
        self.vertices = np.empty((self.N * 2 - 1, self.N * 2 - 1), dtype=np.int32)
        self.lines = np.empty([num_dirs, self.N * 2 - 1, self.N * 2 - 1], dtype=np.int32)
        self.set_initial_board(self.opening)
        self.compute_line()
        self.move_list = []
        self.score = {Color.Black: 0, Color.White: 0}
        self.color_to_play = Color.Black

    def set_initial_board(self, opening):
        if (opening == 0):
            self.vertices.fill(Color.Empty)
            for i in range(0, self.N):
                self.vertices[(0, i)] = Color.White
                self.vertices[(2*self.N - 2, i)] = Color.Black
            for i in range(0, self.N + 1):
                self.vertices[(1, i)] = Color.White
                self.vertices[(2*self.N - 3, i)] = Color.Black
            for i in range(0, 3):
                self.vertices[(2, 2 + i)] = Color.White
                self.vertices[(2 * self.N - 4, 2 + i)] = Color.Black

    def __getitem__(self, index):
        return self.vertices[index]

    def is_on_board(self, x, y):
        d = abs(x - self.N)
        if (d >= self.N):
            return False
        return y >= 0 && y < 2 * self.N - 1 - d

    def compute_line(self):
        for d in range(0, num_dirs):
            for x in range(0, 2 * self.N - 1):
                for y in range(0, 2 * self.N - 1):
                    self.lines[d, x, y] = 0
                    (tx, ty) = (x, y)
                    while self.is_on_board(tx, ty):
                        if (self.vertices[tx, ty] == self.vertices[x, y] and self.vertices[x, y] != Color.Empty):
                            self.lines[d, x, y] += 1
                            tx += dirs[num_dirs - 1 - d][0]
                            ty += dirs[num_dirs - 1 - d][1]
                        else:
                            break



    def adj_vertices(self, xy):
        x,y = xy
        for dx,dy in dirs:
            adj_x,adj_y = x+dx,y+dy
            if self.is_on_board(adj_x, adj_y):
                yield adj_x,adj_y


    def try_make_move(self, x, y, len, line_dir, move_dir, color, actually_execute):
        assert color == Color.White or color == Color.Black

        if (not self.is_on_board(x, y)): return False
        if (self.vertices[x, y] != color): return False

        if (self.lines[line_dir, x, y] < len): return False

        if (len > 3): return False

        move = False

        if (line_dir != move_dir):
            for i in range(0, len):
                nx = x + i * dirs[line_dir][0] + dirs[move_dir][0]
                ny = y + i * dirs[line_dir][1] + dirs[move_dir][1]
                if (not self.vertices[nx, ny] == Color.Empty):
                    return False

            if actually_execute:
                # Acutal move
                for i in range(0, len):
                    self.vertices[x + i * dirs[line_dir][0], y + i * dirs[line_dir][1]] = Color.Empty
                for i in range(0, len):
                    nx = x + i * dirs[line_dir][0] + dirs[move_dir][0]
                    ny = y + i * dirs[line_dir][1] + dirs[move_dir][1]
                    self.vertices[nx, ny] = color
        else:
            nx = x + dirs[move_dir][0]
            ny = y + dirs[move_dir][1]
            if (not self.is_on_board(nx, ny)):
                return False

            if (self.vertices[nx, ny] == color):
                return False

            elif (self.vertices[nx, ny] == Color.Empty):
                if actually_execute:
                    # Acutal move
                    self.vertices[x + (len - 1) * dirs[move_dir][0], y + (len - 1) * dirs[move_dir][1]] = Color.Empty
                    self.vertices[nx, ny] = color

            elif (self.vertices[nx, ny] == flipped_color[color]):

                opp_len = self.lines[num_dirs - 1 - line_dir, nx, ny]
                if (opp_len >= len):
                    return False

                ux = nx + opp_len * dirs[num_dirs - 1 - line_dir][0]
                uy = ny + opp_len * dirs[num_dirs - 1 - line_dir][1]

                if (not self.is_on_board(ux, uy)):
                    if actually_execute:
                        # Acutal move
                        self.score[color] += 1
                        self.vertices[nx, ny] = color
                        tx = x + (len - 1) * dirs[num_dirs - 1 - line_dir][0]
                        ty = y + (len - 1) * dirs[num_dirs - 1 - line_dir][1]
                        self.vertices[x, y] = Color.Empty

                elif (self.vertices[ux, uy] == Color.Empty):
                    if actually_execute:
                        # Actual move
                        self.vertices[ux, uy] = flipped_color[color]
                        self.vertices[nx, ny] = color
                        tx = x + (len - 1) * dirs[num_dirs - 1 - line_dir][0]
                        ty = y + (len - 1) * dirs[num_dirs - 1 - line_dir][1]
                        self.vertices[x, y] = Color.Empty

                else:
                    return False

        if (actually_execute):
            self.move_list.append([x, y, len, line_dir, move_dir])
            self.color_to_play = flipped_color[color]
            self.compute_line()
        return True

    def make_move(self, x, y, len, line_dir, move_dir, color):
        if not self.try_make_move(x, y, len, line_dir, move_dir, color, actually_execute=True):
            raise IllegalMoveException("%s player (%d, %d, len=%d dir=%d, mvdir=%d) is illegal" % (color_names[color], x, y, len, line_dir, move_dir))

    def play_is_legal(self, x, y, len, line_dir, move_dir, color):
        return self.try_make_move(self, x, y, len, line_dir, move_dir, color, actually_execute=False)


    #def flip_colors(self):
    #    for x in range(self.N):
    #        for y in range(self.N):
    #            self.vertices[x,y] = flipped_color[self.vertices[x,y]]

    def show(self):
        color_strings = {
                Color.Empty: '.',
                Color.Black: '\033[31m0\033[0m',
                Color.White: '\033[37m0\033[0m' }
        for x in range(self.N): print "=",
        print
        for y in range(self.N):
            for x in range(self.N):
                if (x,y) == self.simple_ko_vertex:
                    print 'x',
                else:
                    print color_strings[self.vertices[x,y]],
            print
        for x in range(self.N): print "=",
        print

    def show_liberty_counts(self):
        color_strings = {
                Color.Empty: ' .',
                Color.Black: '\033[31m%2d\033[0m',
                Color.White: '\033[37m%2d\033[0m' }
        for x in range(self.N): print " =",
        print
        for y in range(self.N):
            for x in range(self.N):
                s = color_strings[self.vertices[x,y]]
                if self.vertices[x,y] != Color.Empty:
                    s = s % len(self.groups[(x,y)].liberties)
                print s,
            print
        for x in range(self.N): print " =",
        print


def show_sequence(board, moves, first_color):
    board.clear()
    color = first_color
    for x,y in moves:
        legal = board.play_stone(x, y, color)
        board.show()
        color = flipped_color[color]


def test_Board():
    board = Board(5)

    print "simplest capture:"
    show_sequence(board, [(1, 0), (0, 0), (0, 1)], Color.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Color.White)
    board.flip_colors()

    print "bigger capture:"
    show_sequence(board, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)], Color.Black)

    print "ko:"
    show_sequence(board, [(0, 1), (3, 1), (1, 0), (2, 0), (1, 2), (2, 2), (2, 1), (1, 1)], Color.Black)
    print "move at (2, 1) is legal?", board.play_is_legal(2, 1, Color.Black)
    board.show()
    board.flip_colors()
    print "fipped board:"
    board.show()

    print "self capture:"
    show_sequence(board, [(0, 1), (1, 1), (1, 0)], Color.Black)
    print "move at (0, 0) is legal?", board.play_is_legal(0, 0, Color.White)

    print "biffer self capture:"
    show_sequence(board, [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (1, 3), (0, 3), (1, 4)], Color.Black)
    print "move at (0, 4) is legal?", board.play_is_legal(0, 0, Color.White)



if __name__ == "__main__":
    test_Board()



