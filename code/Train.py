#!/usr/bin/python

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import os
import random
import time
from datetime import datetime
import gc
import NNModels
#import MoveTraining
#import InfluenceModels
#import InfluenceTraining
#import EvalModels
#import EvalTraining
#import NPZ
#import Normalization
import Checkpoint
import Selfplay
from Engine import *
from NNEngine import *
from Board import *

save_size = 150

def train_step(total_loss, learning_rate, outcome_op, grad_cap):
    #return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(total_loss)

    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_vars = opt.compute_gradients(total_loss, tf.trainable_variables())
    outcome = tf.reduce_mean(outcome_op)

    dir_grads_vars = [(tf.multiply(tf.clip_by_value(grad, -grad_cap, grad_cap), outcome), var) for grad, var in grads_vars]


    #dir_grads_vars = [[tf.multiply(gv[0], outcome), gv[1]] for gv in grads_vars]

    return opt.apply_gradients(dir_grads_vars), grads_vars, dir_grads_vars


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def read_float_from_file(filename, default):
    try:
        with open(filename, 'r') as f:
            x = float(f.read().strip())
            return x
    except:
        print
        "failed to read from file", filename, "; using default value", default
        return default


def append_line_to_file(filename, s):
    with open(filename, 'a') as f:
        f.write(s)
        f.write('\n')


class MovingAverage:
    def __init__(self, name, time_constant):
        self.name = name
        self.time_constant = time_constant
        self.num_samples = 0
        self.avg = 0.0
        self.last_sample = 0

    def add(self, sample):
        sample = float(sample)
        self.num_samples += 1
        weight = 1.0 / min(self.num_samples, self.time_constant)
        self.avg = weight * sample + (1 - weight) * self.avg
        self.last_sample = sample

    def write(self, summary_writer, step):
        summary_writer.add_summary(make_summary(self.name + ' (avg)', self.avg), step)
        summary_writer.add_summary(make_summary(self.name + ' (raw)', self.last_sample), step)



def loss_func(output_op):
    label_op = tf.placeholder(tf.int64, shape=[None])
    outcome_op = tf.placeholder(tf.float32, shape=[None])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_op, labels=label_op)
    cross_entropy_err = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return cross_entropy_err, label_op, outcome_op


def run_validation(step, engine_strength):
    # play 100 games with different level of AI / previous version NN
    print("Run validation")
    win = 0
    nnengine = NNEngine("NNEngine", model, step)
    stronger_engine = StrongerEngine(model.N, engine_strength)
    num_games = 50

    for i in range(num_games):
        nnengine.clear_board()
        stronger_engine.clear_board()
        if (i % 2 == 0):
            board = Selfplay.run_self_play_game(nnengine, stronger_engine)
            if (board.score[Color.Black] > board.score[Color.White]):
                win = win + 1
        else:
            board = Selfplay.run_self_play_game(stronger_engine, nnengine)
            if (board.score[Color.Black] < board.score[Color.White]):
                win = win + 1
    print("Play against %s: %d/%d" % (stronger_engine.name(), win, num_games))

    return (win / num_games)


def create_self_play_data(model, feature_planes, label_op, outcome_op, list_prv_model):



    feed_dict = []
    cur_step = int(list_prv_model[-1].split('/')[-1].split('-')[-1])
    nnengine = NNEngine("NNEngine", model, cur_step)
    num_prev_version = len(list_prv_model)
    probs = np.empty(num_prev_version)
    probs.fill(1.0 / num_prev_version)
    ind = sample_from(probs)

    m_step = int(list_prv_model[ind].split('/')[-1].split('-')[-1])

    nnengine2 = NNEngine("NNEngine", model, m_step)
    board1 = Selfplay.run_self_play_game(nnengine, nnengine2)

    nnengine.clear_board()
    nnengine2.clear_board()
    board2 = Selfplay.run_self_play_game(nnengine2, nnengine)

    def get_feed(board, color):
        feed = []
        if (board.score[color] > board.score[flipped_color[color]]):
            win = 1
        else:
            win = -1

        temp = Board(model.N)
        for x,y,dir in board.move_list:
            if temp.color_to_play == color:

                feed.append({feature_planes: [Features.make_feature_planes_1(temp, color)],
                                  label_op: [x*(temp.N+1)*temp.move_dirs + y*temp.move_dirs + dir],
                                  outcome_op: [win]})
            temp.make_move(x, y, dir, temp.color_to_play)

        return feed

    return get_feed(board1, Color.Black) + get_feed(board2, Color.White)





def train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, engine_strength, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        grad_cap = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N + 1, N + 1, Nfeat])

        model_outputs = model.inference(feature_planes, N + 1, Nfeat)
        cross_entropy_err, label_op, outcome_op = loss_func(model_outputs)
        train_op, grads_vars_op, dir_grads_vars_op = train_step(cross_entropy_err, learning_rate_ph, outcome_op, grad_cap)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=50, keep_checkpoint_every_n_hours=0.5)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)


        #summary_writer = tf.train.SummaryWriter(
        #    os.path.join(model.train_dir, 'summaries', datetime.now().strftime('%Y%m%d-%H%M%S')), graph=sess.graph,
        #    flush_secs=5)
        #accuracy_avg = MovingAverage('accuracy', time_constant=1000)
        #total_loss_avg = MovingAverage('total_loss', time_constant=1000)



        last_training_loss = None

        if just_validate:  # Just run the validation set once
            step = Checkpoint.restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation(step, engine_strength)
        else:  # Run the training loop

            # step = 0
            step = Checkpoint.optionally_restore_from_checkpoint(sess, saver, os.path.join(model.train_dir))
            saver.save(sess, os.path.join(model.train_dir, "model.ckpt"), global_step=step)

                #saver.save(sess, "/home/alanc/PycharmProjects/Fence/checkpoints/model.ckpt", global_step=0)
            # step = optionally_restore_from_checkpoint(sess, saver, model.train_dir)
            # print "WARNING: CHECKPOINTS TURNED OFF!!"
            print("WARNING: WILL STOP AFTER %d STEPS" % max_steps)
            print("lr_base = %f, lr_half_life = %f" % (lr_base, lr_half_life))
            # loader = NPZ.AsyncRandomizingLoader(train_data_dir, minibatch_size=128)

            print("Step = ", step)


            # loader = NPZ.RandomizingLoader(train_data_dir, minibatch_size=128)
            # loader = NPZ.GroupingRandomizingLoader(train_data_dir, Ngroup=1)
            # loader = NPZ.SplittingRandomizingLoader(train_data_dir, Nsplit=2)
            last_step_ref_time = 0
            while True:


                #print("List of ckpts!!: ", saver.last_checkpoints)

                if step % (2 * save_size) == 0 and step != 0:
                    win_rate = run_validation(step, engine_strength)
                    if (win_rate > 0.95) and (engine_strength >= 0.05):
                        engine_strength = engine_strength - 0.05

                if (step % 10 == 0):
                    print("Step = ", step)

                start_time = time.time()
                # feed_dict = build_feed_dict(loader, normalization, feature_planes, outputs_ph)


                #feed_dict = batch_queue.next_feed_dict()

                # feed_dict = self-play games + make label array
                feed_dict = create_self_play_data(model, feature_planes, label_op, outcome_op, saver.last_checkpoints)

                load_time = time.time() - start_time

                if step % 1 == 0:
                    # learning_rate = read_float_from_file('../work/lr.txt', default=0.1)
                    # momentum = read_float_from_file('../work/momentum.txt', default=0.9)
                    if step < 100:
                        learning_rate = 0.0003  # to stabilize initially
                    else:
                        learning_rate = lr_base * 0.5 ** (float(step - 100) / lr_half_life)
                    momentum = 0.9
                    #summary_writer.add_summary(make_summary('learningrate', learning_rate), step)
                    #summary_writer.add_summary(make_summary('momentum', momentum), step)

                start_time = time.time()

                for feed in feed_dict:
                    feed[learning_rate_ph] = learning_rate
                    feed[grad_cap] = learning_rate * 0.5
                    _, loss_value= sess.run(
                                            [train_op, cross_entropy_err],
                                            feed_dict=feed)


                # feed_dict[learning_rate_ph] = learning_rate
                # feed_dict[momentum_ph] = momentum
                #
                # _, loss_value, outputs_value, grads_vars, dir_grads_vars, outcome = sess.run([train_op, cross_entropy_err, model_outputs,
                #                                          grads_vars_op, dir_grads_vars_op, outcome_op],
                #                                                             feed_dict=feed_dict)
                train_time = time.time() - start_time

                # print("Loss: ", loss_value)
                # print("Grad vars: ", np.shape(grads_vars))
                #print("Max dir grad vars: ", max(dir_grads_vars))
                # print("Outcome: ", outcome)
                #
                # return

                if np.isnan(loss_value):
                    print("Model diverged with loss = Nan")
                    #return
                # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step >= max_steps:
                    return

                #if step % 10 == 0:
                #    total_loss_avg.write(summary_writer, step)
                #    accuracy_avg.write(summary_writer, step)

                full_step_time = time.time() - last_step_ref_time
                last_step_ref_time = time.time()

                if step % 10 == 0:
                    #minibatch_size = feed_dict[feature_planes].shape[0]
                    #examples_per_sec = minibatch_size / full_step_time
                    print("%s: step %d, lr=%.6f, gd_cp=%.6f, loss = %.4f, (load=%.3f train=%.3f total=%0.3f sec/step)" % \
                    (datetime.now(), step, learning_rate, learning_rate * 0.5, loss_value,
                        load_time, train_time, full_step_time))
                    #print("Output values: ", np.array(outputs_value).shape, outputs_value)
                    #if step % 10 == 0:
                        #summary_writer.add_summary(make_summary('examples/sec', examples_per_sec), step)
                        #summary_writer.add_summary(make_summary('step', step), step)

                step += 1

                if step % save_size == 0 and step != 0:
                    # print "WARNING: CHECKPOINTS TURNED OFF!!"
                    saver.save(sess, os.path.join(model.train_dir, "model.ckpt"), global_step=step)
                    #saver.set_last_checkpoints(last_checkpoints=saver.last_checkpoints + [last_check_points])





if __name__ == "__main__":
    N = 5
    # Nfeat = 15
    # Nfeat = 21
    Nfeat = 9

    """
    #model = Models.Conv6PosDep(N, Nfeat) 
    #model = Models.Conv8PosDep(N, Nfeat) 
    #model = Models.Conv10PosDep(N, Nfeat) 
    #model = Models.Conv10PosDepELU(N, Nfeat) 
    #model = MoveModels.Conv12PosDepELU(N, Nfeat) 
    model = MoveModels.Conv12PosDepELUBig(N, Nfeat) 
    #model = MoveModels.Conv16PosDepELU(N, Nfeat) 
    #model = MoveModels.Res5x2PreELU(N, Nfeat) 
    #model = MoveModels.Res10x2PreELU(N, Nfeat) 
    #model = MoveModels.Conv4PosDepELU(N, Nfeat) 
    #model = Models.FirstMoveTest(N, Nfeat) 
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/train-rand-2"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15/val-small"
    #normalization = Normalization.apply_featurewise_normalization_B
    train_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/val-small"
    normalization = Normalization.apply_featurewise_normalization_C
    build_feed_dict = MoveTraining.build_feed_dict
    loss_func = MoveTraining.loss_func
    """

    # model = EvalModels.Conv5PosDepFC1ELU(N, Nfeat)
    model = NNModels.Conv5PosDepELU(N, Nfeat)
    # model = EvalModels.Zero(N, Nfeat)
    # model = EvalModels.Linear(N, Nfeat)
    # train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_Nf21/train"
    #train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_komi_Nf22/train"
    # val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_Nf21/val-small"
    #val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/eval_examples/stones_4lib_4hist_ko_4cap_komi_Nf22/val-small"
    # normalization = Normalization.apply_featurewise_normalization_C
    #normalization = Normalization.apply_featurewise_normalization_D
    #build_feed_dict = EvalTraining.build_feed_dict
    #loss_func = EvalTraining.loss_func

    """
    #model = InfluenceModels.Conv4PosDep(N, Nfeat)
    model = InfluenceModels.Conv12PosDepELU(N, Nfeat)
    train_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/train"
    val_data_dir = "/home/greg/coding/ML/go/NN/data/KGS/influence/examples/stones_3lib_4hist_ko_Nf15/val"
    build_feed_dict = InfluenceTraining.build_feed_dict
    loss_func = InfluenceTraining.loss_func
    normalization = Normalization.apply_featurewise_normalization_B
    """

    # gc.set_debug(gc.DEBUG_STATS)

    #print
    #"Training data = %s\nValidation data = %s" % (train_data_dir, val_data_dir)

    # for lr_half_life in [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]:
    # for lr_half_life in [1e4, 3e4, 1e5, 3e5, 1e6]:
    #    max_steps = lr_half_life * 7
    #    #for lr_base in [0.01, 0.003, 0.001, 0.0003]:
    #    #lr_base = 0.008
    #    lr_base = 0.002 # seems to be the highest useful learning rate for eval_conv11fc1
    #lr_base = 0.001
    lr_base = 0.00006
    lr_half_life = 1e5  # 3e4
    max_steps = 1e9
    engine_strength = 1.0
    train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, engine_strength, just_validate=False)

