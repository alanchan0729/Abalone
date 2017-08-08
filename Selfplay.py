from Board import Board, Color
from Engine import IdiotEngine, StrongerEngine

# Self play games as used by DeepMind to train AlphaGo's value network. Play a
# policy against itself, but insert single random move somewhere in the game.
# Use the position immediately after the random move together with the final
# game result as a single training example for the value network.

def run_self_play_game_with_random_move(engine1, engine2, N=5):
    board = Board(N)

    engine1.set_board_size(N)
    engine1.clear_board()

    engine2.set_board_size(N)
    engine2.clear_board()

    while (board.check_game_end() == False):

        (x, y, d, color) = (-1, -1, -1, -1)

        if (board.color_to_play == Color.Black):
            (x, y, d, color) = engine1.pick_move(board.color_to_play)
        else:
            (x, y, d, color) = engine2.pick_move(board.color_to_play)

        print(x, y, d, color)
        board.make_move(x, y, d, color)
        engine1.move_played(x, y, d, color)
        engine2.move_played(x, y, d, color)
        board.show()

    #print("Scores: ", board.score[Color.Black], board.score[Color.White])
    return (board, board.score[Color.Black], board.score[Color.White])


def test_self_play():
    engine1 = IdiotEngine()
    engine2 = StrongerEngine()

    (board, s1, s2) = run_self_play_game_with_random_move(engine1, engine2)
    print("Final scores: ", s1, s2)



if __name__ == "__main__":
    test_self_play()





