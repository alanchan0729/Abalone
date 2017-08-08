import tensorflow as tf
import numpy as np
import random
import os
from Engine import *
#import Book
import Features
#import Normalization
#import Symmetry
import Checkpoint
#from GTP import Move, true_stderr
from Board import *

def softmax(E, temp):
    #print "E =\n", E
    expE = np.exp(temp * (E - max(E))) # subtract max to avoid overflow
    return expE / np.sum(expE)

def sample_from(probs):
    cumsum = np.cumsum(probs)
    r = random.random()
    #print("cum: ", cumsum)
    #print("r = ", r)
    for i in range(len(probs)):
        if r <= cumsum[i]:
            return i
    print("Prob2: ", probs)
    assert False, "problem with sample_from"


class NNEngine(BaseEngine):
    def __init__(self, eng_name, model, step=None):
        super(NNEngine,self).__init__(model.N)
        self.eng_name = eng_name
        self.model = model
        self.move_records = []
        self.move_probs = []

        # build the graph
        with tf.Graph().as_default():
            #with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N + 1, self.model.N + 1, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N + 1, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.global_variables_initializer()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir)
                Checkpoint.restore_from_checkpoint(self.sess, saver, checkpoint_dir, step)

    def clear_board(self):
        super(NNEngine, self).clear_board()
        self.move_records = []
        self.move_probs = []

    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def set_board_size(self, N):
        if N != self.model.N:
            return False
        return BaseEngine.set_board_size(self, N)

    def pick_model_move(self, color):


        if (self.model.Nfeat == 9):
            board_feature_planes = Features.make_feature_planes_1(self.board, color)
        else:
            assert False


        #feature_batch = Symmetry.make_symmetry_batch(board_feature_planes)

        feed_dict = {self.feature_planes: [board_feature_planes]}

        logit_batch = self.sess.run(self.logits, feed_dict)
        #move_logits = Symmetry.average_plane_over_symmetries(logit_batch, self.model.N)
        move_logits = logit_batch.reshape(((self.model.N+1)*(self.model.N+1)*self.board.move_dirs))

        move_logits = softmax(move_logits, 1.0)

        #if (self.board.empty_edges == 8):
        #    print("Move 1:\n", move_logits)

        # zero out illegal moves
        for x in range(self.model.N + 1):
            for y in range(self.model.N + 1):
                for d in range(self.board.move_dirs):
                    ind = (self.model.N + 1) * self.board.move_dirs * x + y * self.board.move_dirs + d
                    if not self.board.play_is_legal(x, y, d, color):
                        move_logits[ind] = 0

        #print("move probs = ", move_probs)
        sum_probs = np.sum(move_logits)

        if (sum_probs > (1e-10)):

            #if sum_probs == 0: return Move.Pass() # no legal moves, pass
            move_logits /= sum_probs # re-normalize probabilities
            #if (self.board.empty_edges == 8):
            #    print("Move 2:\n", move_logits)

            pick_best = False
            move_x, move_y, move_d = None, None, None

            while (True):
                if pick_best:
                    move_ind = np.argmax(move_logits)
                else:
                    move_ind = sample_from(move_logits)
                move_d = move_ind % self.board.move_dirs
                move_y = (move_ind // self.board.move_dirs) % (self.model.N + 1)
                move_x = (move_ind // self.board.move_dirs) // (self.model.N + 1)
                if (self.board.play_is_legal(move_x, move_y, move_d, color) == True):
                    break

            #print("Ind = ", move_ind, move_x, move_y, move_d)

        else:
            moves = []
            for x in range(self.model.N + 1):
                for y in range(self.model.N + 1):
                    for d in range(self.board.move_dirs):
                        if self.board.play_is_legal(x, y, d, color):
                            moves.append([x, y, d])
            move_x, move_y, move_d = random.choice(moves)

        self.move_records.append([move_x, move_y, move_d])

        return (move_x, move_y, move_d, color)

    def pick_move(self, color):
        return self.pick_model_move(color)

    def get_last_move_probs(self):
        return self.move_probs

    def move_played(self, x, y, dir, color):
        # if we are in kibitz mode, we want to compute model probabilities for ALL turns
        #if self.kibitz_mode:
        #    self.pick_model_move(color)
        #    true_stderr.write("probability of played move %s (%d, %d) was %.2f%%\n" % (color_names[color], x, y, 100*self.last_move_probs[x,y]))

        BaseEngine.move_played(self, x, y, dir, color)




