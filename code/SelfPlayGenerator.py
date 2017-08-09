
import multiprocessing
import random
import Features
from Board import *
from NNEngine import *


def run_self_play_game(engine1, engine2, show=False):
    board = Board(engine1.board.N)

    engine1.clear_board()
    engine2.clear_board()

    while (board.check_game_end() == False):

        (x, y, d, color) = (-1, -1, -1, -1)
        if (board.color_to_play == Color.Black):
            (x, y, d, color) = engine1.pick_move(board.color_to_play)
        else:
            (x, y, d, color) = engine2.pick_move(board.color_to_play)

        #print("Selfplay: ", x, y, d, color)
        board.make_move(x, y, d, color)
        engine1.move_played(x, y, d, color)
        engine2.move_played(x, y, d, color)
        if (show == True):
            print(x, y, d, color)
            board.show()

    #print("Scores: ", board.score[Color.Black], board.score[Color.White])
    return board

def get_feed(board, color, feature_planes_ph, label_ph, outcome_ph):
    feed = []
    if (board.score[color] > board.score[flipped_color[color]]):
        win = 1
    else:
        win = -1

    temp = Board(board.N)
    for x,y,dir in board.move_list:
        if temp.color_to_play == color:

            feed.append({feature_planes_ph: [Features.make_feature_planes_1(temp, color)],
                              label_ph: [x*(temp.N+1)*temp.move_dirs + y*temp.move_dirs + dir],
                              outcome_ph: [win]})
        temp.make_move(x, y, dir, temp.color_to_play)

    return feed


def gameplay_worker(engines, main_engine, minibatch_size, feature_planes_ph, label_ph, outcome_ph):

    feed_dict = []

    for i in range(minibatch_size):
        ind = random.randint(0, len(engines) - 1)


        board1 = run_self_play_game(main_engine, engines[ind])


        board2 = run_self_play_game(engines[ind], main_engine)

        f1 = get_feed(board1, Color.Black, feature_planes_ph, label_ph, outcome_ph)
        feed_dict += f1

        f2 = get_feed(board2, Color.White, feature_planes_ph, label_ph, outcome_ph)
        feed_dict += f2

    return feed_dict


class AsyncRandomGamePlayQueue:

    def __init__(self):
        self.engines = []
        self.versions = []
        #self.num_process = 3

    def start_gen_game_play(self, feature_planes_ph, label_ph, outcome_ph, minibatch_size, versions, model):

        #self.q = multiprocessing.Queue(maxsize=300)
        self.counter = 0
        self.main_engine = NNEngine("NN Engine %d" % versions[-1], model, versions[-1])

        # update engines
        if (versions[-1] not in self.versions):
            self.versions.append(versions[-1])
            self.engines.append(NNEngine("NN Engine %d" % versions[-1], model, versions[-1]))

        if (self.versions[0] not in versions):
            self.versions.pop(0)
            self.engines.pop(0)

        # create process
        #self.process = multiprocessing.Process(target=gameplay_worker, args=(self.q, self.engines, self.main_engine, minibatch_size))
        self.feed_dict = gameplay_worker(self.engines, self.main_engine, minibatch_size, feature_planes_ph, label_ph, outcome_ph)



    def next_feed_dict(self):
        if self.counter < len(self.feed_dict):
            self.counter += 1
            return self.feed_dict[self.counter - 1]
        else:
            return None