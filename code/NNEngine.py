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
    for i in range(len(probs)):
        if r <= cumsum[i]:
            return i
    assert False, "problem with sample_from"


class NNEngine(BaseEngine):
    def __init__(self, eng_name, model, step=None):
        super(NNEngine,self).__init__()
        self.eng_name = eng_name
        self.model = model
        self.move_records = []
        self.move_probs = []

        # build the graph
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                self.feature_planes = tf.placeholder(tf.float32, shape=[None, self.model.N, self.model.N, self.model.Nfeat], name='feature_planes')
                self.logits = model.inference(self.feature_planes, self.model.N, self.model.Nfeat)
                saver = tf.train.Saver(tf.trainable_variables())
                init = tf.initialize_all_variables()
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
                self.sess.run(init)
                checkpoint_dir = os.path.join(model.train_dir, 'checkpoints')
                Checkpoint.restore_from_checkpoint(self.sess, saver, checkpoint_dir, step)


    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def set_board_size(self, N):
        if N != self.model.N:
            return False
        return BaseEngine.set_board_size(self, N)

    def pick_model_move(self, color):

        if (self.model.Nfeat == 11):
            board_feature_planes = Features.make_feature_planes_1(self.board, color)
        else:
            assert False


        #feature_batch = Symmetry.make_symmetry_batch(board_feature_planes)

        feed_dict = {self.feature_planes: board_feature_planes}

        logit_batch = self.sess.run(self.logits, feed_dict)
        #move_logits = Symmetry.average_plane_over_symmetries(logit_batch, self.model.N)
        move_logits = logit_batch.reshape((self.model.N*self.model.N*self.board.num_dirs))
        softmax_temp = 1.0
        move_probs = softmax(move_logits, softmax_temp)

        # zero out illegal moves
        for x in range(self.model.N):
            for y in range(self.model.N):
                ind = self.model.N * x + y
                if not self.board.play_is_legal(x, y, color):
                    move_probs[ind] = 0
        sum_probs = np.sum(move_probs)

        #if sum_probs == 0: return Move.Pass() # no legal moves, pass
        move_probs /= sum_probs # re-normalize probabilities

        pick_best = False
        if pick_best:
            move_ind = np.argmax(move_probs)
        else:
            move_ind = sample_from(move_probs)

        move_d = move_ind % self.board.num_dirs
        move_y = (move_ind/self.board.num_dirs) % self.model.N
        move_x = (move_ind/self.board.num_dirs) / self.model.N

        self.move_records.append([move_x, move_y, move_d])

        self.move_probs.append(move_probs.reshape((self.board.N, self.board.N, self.board.num_dirs)))

        return (move_x, move_y, move_d)

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




