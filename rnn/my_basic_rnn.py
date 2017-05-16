import tensorflow as tf
import numpy as np

from mylibs.jupyter_notebook_helper import getRunTime, initStats, gatherStats
from mylibs.tf_helper import tfRMSE, tfMSE, fully_connected_layer, trainEpoch, validateEpoch
from msd10_data_providers import MSD10Genre_120_rnn_native_DataProvider


class MyBasicRNN(object):
    num_classes = 10

    def __init__(self, batch_size, rng, dtype, config, segment_count, segment_len):
        super(MyBasicRNN, self).__init__()

        self.batch_size = batch_size
        self.rng = rng
        self.dtype = dtype
        self.config = config

        self.train_data = None
        self.valid_data = None

        self.init = None
        self.error = None
        self.accuracy = None
        self.inputs = None
        self.targets = None
        # self.training = None
        # self.keep_prob_input = None
        # self.keep_prob_hidden = None
        self.train_step = None
        self.logits = None

        self.segment_count = segment_count
        self.segment_len = segment_len

    def get_dp(self, which_set, num_steps):
        return MSD10Genre_120_rnn_native_DataProvider(
            num_steps=num_steps, which_set=which_set, batch_size=self.batch_size, rng=self.rng
        )

    def loadAndGetDataProviders(self, num_steps):
        self.train_data = self.get_dp('train', num_steps)
        self.valid_data = self.get_dp('valid', num_steps)

        return self.train_data, self.valid_data

    def loadAndGetGraph(self, num_steps, state_size,
                        learningRate=1e-4  # default of Adam is 1e-3
                        ):
        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(
                    self.dtype,
                    [self.batch_size, self.segment_count, self.segment_len],
                    name='input_placeholder'
                )

                targets = tf.placeholder(self.dtype, [self.batch_size, self.num_classes],
                                         name='labels_placeholder')

                init_state = tf.zeros([self.batch_size, state_size], dtype=self.dtype)

            # list where each item have dim 50 x 25
            rnn_inputs = tf.unpack(inputs, axis=1, name='rnn_inputs')

            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)  # tanh is default activation

            rnn_outputs, final_state = tf.nn.rnn(
                cell, rnn_inputs,
                initial_state=init_state, sequence_length=np.repeat(num_steps, self.batch_size)
            )

            with tf.variable_scope('readout'):
                input_dim = state_size
                output_dim = self.num_classes
                W = tf.Variable(
                    tf.truncated_normal(
                        [input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5
                    ),
                    name='readout_weights'
                )
                b = tf.Variable(tf.zeros([output_dim]),
                                name='readout_biases')

                logits = tf.matmul(final_state, W) + b  # shape: (50, 10)

            # print logits.get_shape()

            with tf.name_scope('error'):
                error = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, targets)
                )

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),
                                                           tf.argmax(targets, 1)),
                                                  dtype=self.dtype))

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.error = error
        self.accuracy = accuracy
        self.inputs = inputs
        self.targets = targets
        self.train_step = train_step
        self.logits = logits

        return graph

    def rnn_native_tf(self, state_size, num_steps, epochs):
        train_data, valid_data = self.loadAndGetDataProviders(num_steps=num_steps)

        graph = self.loadAndGetGraph(num_steps=num_steps, state_size=state_size)

        stats, keys = self.trainAndValidate(
            train_data, valid_data, state_size, graph, num_steps = num_steps, epochs=epochs
        )

        return stats, keys

    def trainAndValidate(self, train_data, valid_data, state_size, graph, num_steps, epochs=35, verbose=True):
        if verbose:
            print "epochs: %d" % epochs
            print "rnn steps: %d" % num_steps
            print "state size: %d" % state_size

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            stats, keys = initStats(epochs)

            for e in range(epochs):
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda:
                    trainEpoch(
                        inputs=self.inputs, targets=self.targets, sess = sess, train_data= train_data,
                        train_step=self.train_step, error=self.error, accuracy=self.accuracy
                    )
                )

                if (e + 1) % 1 == 0:
                    valid_error, valid_accuracy = validateEpoch(
                        inputs=self.inputs, targets=self.targets, sess=sess,
                        valid_data=valid_data, error=self.error, accuracy=self.accuracy
                    )

                if verbose:
                    print 'End epoch %02d (%.3f secs): err(train)=%.2f, acc(train)=%.2f, err(valid)=%.2f, acc(valid)=%.2f, ' % \
                          (e + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)

                stats = gatherStats(e, train_error, train_accuracy,
                                    valid_error, valid_accuracy, stats)

        if verbose:
            print

        return stats, keys
