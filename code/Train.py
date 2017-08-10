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
import SelfPlayGenerator


def train_step(total_loss, learning_rate, outcome_op, grad_cap):
    #return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(total_loss)

    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_vars = opt.compute_gradients(total_loss, tf.trainable_variables())
    outcome = tf.reduce_mean(outcome_op)

    #dir_grads_vars = [(tf.multiply(tf.clip_by_value(grad, -grad_cap, grad_cap), outcome), var) for grad, var in grads_vars]
    dir_grads_vars = [(grad * outcome, var) for grad, var in grads_vars]


    #dir_grads_vars = [[tf.multiply(gv[0], outcome), gv[1]] for gv in grads_vars]

    return opt.apply_gradients(dir_grads_vars), grads_vars, dir_grads_vars


def loss_func(output_op, label_op):

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_op, labels=label_op)
    cross_entropy_err = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    return cross_entropy_err


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
        show = False
        #if (i == 30):
        #    show = True

        if (i % 2 == 0):
            board = SelfPlayGenerator.run_self_play_game(nnengine, stronger_engine, show=show, pick_best=True)
            if (board.score[Color.Black] > board.score[Color.White]):
                win = win + 1
        else:
            board = SelfPlayGenerator.run_self_play_game(stronger_engine, nnengine, pick_best=True)
            if (board.score[Color.Black] < board.score[Color.White]):
                win = win + 1
    print("Play against %s: %d/%d" % (stronger_engine.name(), win, num_games))

    return (win / num_games)





def train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, minibatch_size, engine_strength, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        grad_cap = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N + 1, N + 1, Nfeat])

        model_outputs = model.inference(feature_planes, N + 1, Nfeat)
        label_op = tf.placeholder(tf.int64, shape=[None])
        outcome_op = tf.placeholder(tf.float32, shape=[None])
        cross_entropy_err = loss_func(model_outputs, label_op)
        train_op, grads_vars_op, dir_grads_vars_op = train_step(cross_entropy_err, learning_rate_ph, outcome_op, grad_cap)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=800, keep_checkpoint_every_n_hours=0.5)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)


        last_training_loss = None

        if just_validate:  # Just run the validation set once
            step = Checkpoint.restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation(step, engine_strength)
        else:  # Run the training loop

            # step = 0
            step = Checkpoint.optionally_restore_from_checkpoint(sess, saver, os.path.join(model.train_dir))
            saver.save(sess, os.path.join(model.train_dir, "model.ckpt"), global_step=step)


            print("WARNING: WILL STOP AFTER %d STEPS" % max_steps)
            print("lr_base = %f, lr_half_life = %f" % (lr_base, lr_half_life))
            # loader = NPZ.AsyncRandomizingLoader(train_data_dir, minibatch_size=128)

            batch_queue = SelfPlayGenerator.AsyncRandomGamePlayQueue()

            last_step_ref_time = 0
            while True:


                if step % (10 * minibatch_size) == 0 and step != 0:
                    win_rate = run_validation(step, engine_strength)
                    if (win_rate > 0.90) and (engine_strength >= 0.05):
                        engine_strength = engine_strength - 0.05

                start_time = time.time()

                checkpoint_paths = saver.last_checkpoints
                versions = [int(checkpoint_paths[ind].split('/')[-1].split('-')[-1]) for ind in
                            range(len(checkpoint_paths))]

                batch_queue.start_gen_game_play(feature_planes, label_op, outcome_op, minibatch_size, versions, model)


                if step < 100:
                    learning_rate = 0.0003  # to stabilize initially
                else:
                    learning_rate = lr_base * 0.5 ** (float(step - 110000) / lr_half_life)
                #summary_writer.add_summary(make_summary('learningrate', learning_rate), step)
                #summary_writer.add_summary(make_summary('momentum', momentum), step)

                mean_loss = 0.0
                counter = 0

                while True:
                    feed_dict = batch_queue.next_feed_dict()
                    if (feed_dict == None):
                        break
                    feed_dict[learning_rate_ph] = learning_rate
                    feed_dict[grad_cap] = learning_rate * 1.0
                    _, loss_value = sess.run(
                        [train_op, cross_entropy_err],
                        feed_dict=feed_dict)

                    # _, loss_value, model_out, label_out = sess.run(
                    #     [train_op, cross_entropy_err, model_outputs, label_op],
                    #     feed_dict=feed_dict)
                    #
                    # e_x = np.exp(model_out - np.max(model_out))
                    # e_x = e_x / e_x.sum()
                    # print("Model out: \n", e_x)
                    # print("True out: \n", label_out)
                    # print("Loss: ", loss_value)
                    #
                    # if (loss_value > 30):
                    #     return

                    mean_loss += loss_value
                    counter += 1

                mean_loss /= counter

                # _, loss_value, outputs_value, grads_vars, dir_grads_vars, outcome = sess.run([train_op, cross_entropy_err, model_outputs,
                #                                          grads_vars_op, dir_grads_vars_op, outcome_op],
                #                                                             feed_dict=feed_dict)
                train_time = time.time() - start_time

                if step >= max_steps:
                    return

                #if step % 10 == 0:
                #    total_loss_avg.write(summary_writer, step)
                #    accuracy_avg.write(summary_writer, step)

                full_step_time = time.time() - last_step_ref_time
                last_step_ref_time = time.time()

                if step % 10 == 0:
                    print("%s: step %d, lr=%f, loss = %.2f, (train=%.3f s/%d)" % \
                    (datetime.now(), step, learning_rate, mean_loss,
                        train_time, minibatch_size))

                step += minibatch_size

                if step % minibatch_size == 0 and step != 0:
                    saver.save(sess, os.path.join(model.train_dir, "model.ckpt"), global_step=step)





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
    lr_base =  0.0000015
    lr_half_life = 5e5  # 3e4
    max_steps = 1e9
    minibatch_size = 20
    engine_strength = 1.0
    train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, minibatch_size, engine_strength, just_validate=False)

