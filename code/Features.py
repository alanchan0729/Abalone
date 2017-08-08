#!/usr/bin/python
import numpy as np
import os
import os.path
import random
import math
from Board import *


def make_board_plane(array, board):
    np.copyto(array, board.edges)


def make_ones_plane(array, board):
    np.copyto(array, np.ones((board.N, board.N), dtype=np.int8))

def make_score_diff_plane(plane, board, play_color):
    score_diff = board.score[Color.Black] - board.score[Color.White]
    if play_color == Color.White:
        score_diff = -score_diff
    plane.fill(score_diff)

def make_grid_count_planes(array, board):
    for x in range(board.N):
        for y in range(board.N):
            count = board.count_fenced(x, y)
            array[x, y, count] = 1

# us, them, empty, ones

def make_feature_planes_0(board, play_color):
    Nplanes = 6
    feature_planes = np.zeros((board.N, board.N, Nplanes), dtype=np.int8)
    make_board_plane(feature_planes[:, :, 0:4], board)
    make_ones_plane(feature_planes[:, :, 4], board)
    make_score_diff_plane(feature_planes[:, :, 5], board, play_color)
    return feature_planes

def make_feature_planes_1(board, play_color):
    Nplanes = 11
    feature_planes = np.zeros((board.N, board.N, Nplanes), dtype=np.int8)
    make_board_plane(feature_planes[:, :, 0:4], board)
    make_ones_plane(feature_planes[:, :, 4], board)
    make_score_diff_plane(feature_planes[:, :, 5], board, play_color)
    make_grid_count_planes(feature_planes[:, :, 6:11], board)
    return feature_planes



def test_features():
    board = Board(5)
    moves = [(0, 0, 1), (0, 0, 0), (0, 0, 2), (0, 0, 3), (2, 2, 0),(2, 2, 1),(2, 2, 2),(3, 3, 1), (3, 3, 2), (4, 4, 0)]
    for (x, y, d) in moves:
        board.make_move(x, y, d, board.color_to_play)
    f_planes = make_feature_planes_1(board, Color.Black)

    board.show()
    for i in range(11):
        print(f_planes[:, :, i])

if __name__ == "__main__":
    test_features()

