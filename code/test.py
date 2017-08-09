import tensorflow as tf
import numpy as np
import random

# with tf.Graph().as_default():
#     #return a list of trainable variable in you model
#
#
#
#     params = tf.trainable_variables()
#
#     #create an optimizer
#     opt = tf.train.GradientDescentOptimizer(self.learning_rate)
#
#     #compute gradients for params
#     gradients = tf.gradients(loss, params)
#
#     #process gradients
#     clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
#
#     train_op = opt.apply_gradients(zip(clipped_gradients, params)))


with tf.Graph().as_default():
    label = tf.placeholder(tf.int64)
    x = tf.placeholder(tf.float32, shape=[5])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=label)

    ss = tf.Session()

    feed_dict = {x: [1.0, 0.0, 0.0, 0.0, 0.0], label: 0}

    LOSS = ss.run([loss], feed_dict=feed_dict)

    print(LOSS)





training_graph = tf.Graph()

with tf.Graph().as_default():

    a = tf.add(2, 5)
    b = tf.multiply(a, 3)
    sx = tf.Session()

    engine1 = tf.Graph()
    engine2 = tf.Graph()

    with tf.Graph().as_default():
        x1 = tf.add(10, 20)
        y1 = tf.multiply(x1, 7)
        s1 = tf.Session()

    with tf.Graph().as_default():
        x2 = tf.add(15, 25)
        y2 = tf.multiply(x2, 5)
        s2 = tf.Session()


    print(s1.run(y1))
    print(s2.run(y2))
    print(sx.run(b))


v = tf.Variable(tf.constant(0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, 'haha/trained_variables.ckpt', global_step=0)