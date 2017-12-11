import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function = None):
# add one more layer and return the output of this layer
    with tf.name_scope('layer') as scope:
        with tf.name_scope('weights') as scope:
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
        with tf.name_scope('biases') as scope:
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b') as scope:
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

# define placeholder for input to network
with tf.name_scope('inputs') as scope:
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
            reduction_indices=[1]))
with tf.name_scope('train') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
# important step
init = tf.global_variables_initializer()
sess.run(init)
