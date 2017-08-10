
import multiprocessing
import random
import Features
from Board import *
from NNEngine import *
import Engine
import numpy as np


def run_self_play_game(engine1, engine2, show=False, rounds=None, pick_best=False):
    board = Board(engine1.board.N)
    board.clear()

    engine1.clear_board()
    engine2.clear_board()

    counter = 0

    while (board.check_game_end() == False and counter != rounds):

        (x, y, d, color) = (-1, -1, -1, -1)
        if (board.color_to_play == Color.Black):
            (x, y, d, color) = engine1.pick_move(board.color_to_play, show=show, pick_best=pick_best)
        else:
            (x, y, d, color) = engine2.pick_move(board.color_to_play, pick_best=pick_best)

        #print("Selfplay: ", x, y, d, color)
        board.make_move(x, y, d, color)
        engine1.move_played(x, y, d, color)
        engine2.move_played(x, y, d, color)
        if (show == True):
            print(x, y, d, color)
            board.show()

        counter += 1

    #print("Scores: ", board.score[Color.Black], board.score[Color.White])
    return board

def get_feed(board, color):
    feed = []
    if (board.score[color] > board.score[flipped_color[color]]):
        win = 1
    else:
        win = -1

    temp = Board(board.N)
    for x,y,dir in board.move_list:
        if temp.color_to_play == color:

            feed.append({'feature_planes': [Features.make_feature_planes_1(temp, color)],
                              'label': [x*(temp.N+1)*temp.move_dirs + y*temp.move_dirs + dir],
                              'outcome': [win]})
        temp.make_move(x, y, dir, temp.color_to_play)

    return feed


def gameplay_worker(engines, main_engine, minibatch_size):

    feed_dict = []

    for i in range(minibatch_size):
        ind = random.randint(0, len(engines) - 1)

        c = random.randint(0, 1)

        if (c == 0):
            board = run_self_play_game(main_engine, engines[ind])
            feed_dict += get_feed(board, Color.Black)
        else:
            board = run_self_play_game(engines[ind], main_engine)
            feed_dict += get_feed(board, Color.White)

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
        self.feature_planes = feature_planes_ph
        self.label = label_ph
        self.outcome = outcome_ph

        # update engines
        if (versions[-1] not in self.versions):
            self.versions.append(versions[-1])
            self.engines.append(NNEngine("NN Engine %d" % versions[-1], model, versions[-1]))

        if (self.versions[0] not in versions):
            self.versions.pop(0)
            self.engines.pop(0)

        # create process
        #self.process = multiprocessing.Process(target=gameplay_worker, args=(self.q, self.engines, self.main_engine, minibatch_size))
        self.feed_dict = gameplay_worker(self.engines, self.main_engine, minibatch_size)



    def next_feed_dict(self):
        if self.counter < len(self.feed_dict):
            self.counter += 1
            temp = self.feed_dict[self.counter - 1]
            return {self.feature_planes: temp['feature_planes'],
                    self.label: temp['label'],
                    self.outcome: temp['outcome']}
        else:
            return None


def async_gen_board(queue, N):

    while True:

        moves = random.randint(0, 2 * N * (N + 1) - 1)

        idiot1 = Engine.IdiotEngine(N)
        idiot2 = Engine.IdiotEngine(N)

        board = run_self_play_game(idiot1, idiot2, rounds=moves)
        #print("moves = ", moves)
        #board.show()

        true_outputs = np.zeros((N + 1, N + 1, 2), dtype=np.float32)
        for x in range(N+1):
            for y in range(N+1):
                for d in range(2):
                    if (board.play_is_legal(x, y, d, board.color_to_play) == True):
                        true_outputs[x, y, d] = 1

        true_outputs = true_outputs.reshape([-1])
        ##print(sum(true_outputs))
        true_outputs /= sum(true_outputs)


        feed = {'feature_planes': [Features.make_feature_planes_1(board, board.color_to_play)],
                'true_outputs': [true_outputs]}

        queue.put(feed, block=True)


class AsyncRandomBoardQueue:

    def __init__(self, N, feature_planes_ph, true_outputs_ph):
        self.q = multiprocessing.Queue(maxsize=5)
        self.N = N
        self.process = multiprocessing.Process(target=async_gen_board, args=(self.q, self.N))
        self.process.daemon = True
        self.process.start()
        self.feature_planes_ph = feature_planes_ph
        self.true_outputs_ph = true_outputs_ph


    def next_feed_dict(self):
        feed_dict = self.q.get(block=True, timeout=30)
        return {self.feature_planes_ph: feed_dict['feature_planes'],
                self.true_outputs_ph: feed_dict['true_outputs']}