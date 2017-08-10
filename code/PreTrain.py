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


def train_step(total_loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def loss_func(output_op, label_op):

    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_op, labels=label_op)
    #cross_entropy_err = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    #loss = tf.reduce_mean(tf.squared_difference(output_op, label_op))
    output_op = tf.nn.softmax(output_op)

    cross_entropy = -tf.reduce_sum(label_op * tf.log(output_op + 0.00001))
    entropy = -tf.reduce_sum(label_op * tf.log(label_op + 0.00001))
    kl_divergence = cross_entropy - entropy

    return kl_divergence


def train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, just_validate=False):
    with tf.Graph().as_default():
        # build the graph
        learning_rate_ph = tf.placeholder(tf.float32)
        feature_planes = tf.placeholder(tf.float32, shape=[None, N + 1, N + 1, Nfeat])

        model_outputs = model.inference(feature_planes, N + 1, Nfeat)
        true_outputs = tf.placeholder(tf.float32, shape=[None, (N+1)*(N+1)*2])
        loss = loss_func(model_outputs, true_outputs)
        train_op = train_step(loss, learning_rate_ph)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=20, keep_checkpoint_every_n_hours=0.5)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        ## initialize gen board queue
        batch_queue = SelfPlayGenerator.AsyncRandomBoardQueue(model.N, feature_planes, true_outputs)

        last_training_loss = None

        def run_validation(step, model, queue):
            # play 100 games with different level of AI / previous version NN
            print("Run validation")
            num_board = 100
            total_err = 0

            for i in range(num_board):

                feed_dict = queue.next_feed_dict()
                loss_value, model_out, true_out = sess.run([loss, model_outputs, true_outputs], feed_dict=feed_dict)
                total_err += loss_value

                if (i % 50 == 0):
                    e_x = np.exp(model_out - np.max(model_out))
                    e_x = e_x / e_x.sum()
                    print("Model out: \n", e_x)
                    print("True out: \n", true_out)

            total_err /= num_board

            print("Validation error: ", total_err)

        if just_validate:  # Just run the validation set once
            step = Checkpoint.restore_from_checkpoint(sess, saver, model.train_dir)
            run_validation(step, model, batch_queue)
        else:  # Run the training loop

            step = Checkpoint.optionally_restore_from_checkpoint(sess, saver, os.path.join(model.train_dir))
            saver.save(sess, os.path.join(model.train_dir, "model.ckpt"), global_step=step)


            print("WARNING: WILL STOP AFTER %d STEPS" % max_steps)
            print("lr_base = %f, lr_half_life = %f" % (lr_base, lr_half_life))

            last_step_ref_time = 0


            test_size = 2000
            save_size = 500

            while True:

                if step % test_size == 0 and step != 0:
                    run_validation(step, model, batch_queue)

                if step < 100:
                    learning_rate = 0.0003  # to stabilize initially
                else:
                    learning_rate = lr_base * 0.5 ** (float(step - 100) / lr_half_life)

                #if (step % 20 == 0):
                #    print("Step = ", step)

                start_time = time.time()

                feed_dict = batch_queue.next_feed_dict()
                feed_dict[learning_rate_ph] = learning_rate
                _, loss_value = sess.run(
                        [train_op, loss],
                        feed_dict=feed_dict)

                train_time = time.time() - start_time

                #print(model_out)
                #print(true_out)

                if step >= max_steps:
                    return

                if step % 100 == 0:
                    print("%s: step %d, lr=%.6f, loss = %.4f, (train=%.3f sec/ step)" % \
                    (datetime.now(), step, learning_rate, loss_value, train_time))

                step += 1

                if step % save_size == 0 and step != 0:
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
    build_feed_dict = MoveTraining.build_feed_dicttrue_outputs = true_outputs.reshape([-1])
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

    lr_base = 0.01
    lr_half_life = 7e5  # 3e4
    max_steps = 1e9
    engine_strength = 1.0
    train_model(model, N, Nfeat, lr_base,
                lr_half_life, max_steps, just_validate=False)

