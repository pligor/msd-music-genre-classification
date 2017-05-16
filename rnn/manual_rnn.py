import tensorflow as tf
import numpy as np
import os
from mylibs.jupyter_notebook_helper import getRunTime, initStats, gatherStats
from msd10_data_providers import MSD10Genre_120_rnn_DataProvider
from mylibs.py_helper import merge_dicts
from rnn.rnn_model_interface import RnnModelInterface
from rnn.cross_validator import CrossValidator
from rnn.logits_gatherer import LogitsGatherer
from rnn.data_providers import MSD120RnnDataProvider
from rnn.kaggle_rnn import KaggleRNN


class ManualRNN(CrossValidator, RnnModelInterface, LogitsGatherer, KaggleRNN):
    __default_learning_rate = 1e-4

    def __init__(self, batch_size, rng, dtype, config, segment_count, segment_len, seed=16011984, num_classes=10):
        super(ManualRNN, self).__init__(batch_size=batch_size, seed=seed)

        self.batch_size = batch_size
        self.rng = rng
        self.dtype = dtype
        self.config = config
        self.num_classes = num_classes

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
        self.outputs = None
        self.init_state = None
        self.softmax = None

        self.segment_count = segment_count
        self.segment_len = segment_len

    def get_dp(self, which_set, num_steps, shuffle_order=True):
        if self.num_classes == 10:
            return MSD10Genre_120_rnn_DataProvider(
                num_steps=num_steps, which_set=which_set, batch_size=self.batch_size, rng=self.rng,
                shuffle_order=shuffle_order
            )
        else:
            return MSD120RnnDataProvider(
                num_steps=num_steps, which_set=which_set, batch_size=self.batch_size, rng=self.rng,
                num_classes=self.num_classes, shuffle_order=shuffle_order
            )

    def loadAndGetDataProviders(self, num_steps):
        self.train_data = self.get_dp('train', num_steps)
        self.valid_data = self.get_dp('valid', num_steps)

        return self.train_data, self.valid_data

    def run_rnn(self, state_size, num_steps, epochs, kaggleEnabled=False, learning_rate=__default_learning_rate):
        train_data, valid_data = self.loadAndGetDataProviders(num_steps=num_steps)

        return self.run_rnn_for_data(state_size=state_size, num_steps=num_steps, epochs=epochs, train_data=train_data,
                                     valid_data=valid_data, kaggleEnabled=kaggleEnabled, learning_rate=learning_rate)

    def run_rnn_for_data(self, state_size, num_steps, epochs, train_data, valid_data, logits_gathering_enabled=False,
                         kaggleEnabled=False, learning_rate=__default_learning_rate):
        graph = self.getGraph(num_steps=num_steps, state_size=state_size, learningRate=learning_rate)

        if kaggleEnabled:
            assert self.num_classes == 25

        return self.trainAndValidate(
            train_data=train_data, valid_data=valid_data, state_size=state_size, graph=graph, epochs=epochs,
            logits_gathering_enabled=logits_gathering_enabled,
            test_data=self.get_dp('test', num_steps, shuffle_order=False) if kaggleEnabled else None
        )

    def validate(self, data_provider, state_size, graph, epochs=35, verbose=True):
        """onLogits(sess, logits, create_feed_dict, filename = 'msd-25-submission.csv')"""
        if verbose:
            print "epochs: %d" % epochs
            print "rnn steps: %d" % data_provider.num_steps
            print "state size: %d" % state_size

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            stats, keys = initStats(epochs)

            for e in range(epochs):
                (valid_error, valid_accuracy), runTime = getRunTime(
                    lambda: self.validateEpoch(state_size=state_size, sess=sess, valid_data=data_provider)
                )

                if verbose:
                    print 'End epoch %02d (%.3f secs): err(valid)=%.2f, acc(valid)=%.2f, ' % \
                          (e + 1, runTime, valid_error, valid_accuracy)

                stats = gatherStats(e, train_error=0., train_accuracy=0.,
                                    valid_error=valid_error, valid_accuracy=valid_accuracy,
                                    stats=stats)

        if verbose:
            print

        return stats, keys

    def trainAndValidate(self, train_data, valid_data, state_size, graph, epochs=35, verbose=True,
                         logits_gathering_enabled=False, test_data=None):
        """onLogits(sess, logits, create_feed_dict, filename = 'msd-25-submission.csv')"""
        if verbose:
            print "epochs: %d" % epochs
            print "rnn steps: %d" % train_data.num_steps
            print "state size: %d" % state_size

        logits_dict = None
        max_valid_acc = 0.

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            stats, keys = initStats(epochs)

            for e in range(epochs):
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda: self.trainEpoch(state_size=state_size, sess=sess, train_data=train_data)
                )

                # if (e + 1) % 1 == 0: #better calc it always
                valid_error, valid_accuracy = self.validateEpoch(
                    state_size=state_size, sess=sess, valid_data=valid_data
                )

                if valid_accuracy > max_valid_acc:
                    if logits_gathering_enabled:
                        logits_dict, _, _ = self.getLogits(
                            batch_size=self.batch_size, data_provider=valid_data, sess=sess, state_size=state_size
                        )

                    if test_data is not None:
                        self.createKaggleFile(batch_size=self.batch_size, data_provider=test_data, sess=sess,
                                              state_size=state_size)

                    max_valid_acc = valid_accuracy

                if verbose:
                    print 'End epoch %02d (%.3f secs): err(train)=%.2f, acc(train)=%.2f, err(valid)=%.2f, acc(valid)=%.2f, ' % \
                          (e + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)

                stats = gatherStats(e, train_error, train_accuracy,
                                    valid_error, valid_accuracy, stats)

        if verbose:
            print

        if logits_gathering_enabled:
            return stats, keys, logits_dict
        else:
            return stats, keys

    def validateEpoch(self, state_size, sess, valid_data, extraFeedDict={}):
        valid_error = 0.
        valid_accuracy = 0.

        zeroState = lambda: np.zeros([self.batch_size, state_size])

        cur_state = zeroState()

        for step, ((input_batch, target_batch), segmentPartCounter) in enumerate(valid_data):
            cur_state, batch_error, batch_acc = sess.run(
                [self.outputs, self.error, self.accuracy],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.init_state: cur_state}, extraFeedDict)
            )

            if (segmentPartCounter + 1) % valid_data.segment_part_count == 0:
                cur_state = zeroState()
                # only care about the last output of the training error and acc of the rnn
                # so include it in if-statement
                valid_error += batch_error
                valid_accuracy += batch_acc

        num_batches = valid_data.num_batches

        valid_error /= num_batches
        valid_accuracy /= num_batches

        return valid_error, valid_accuracy

    def trainEpoch(self, state_size, sess, train_data, extraFeedDict={}):
        train_error = 0.
        train_accuracy = 0.
        # train_error = []
        # train_accuracy = []

        num_batches = train_data.num_batches

        zeroState = lambda: np.zeros([self.batch_size, state_size])

        cur_state = zeroState()

        for step, ((input_batch, target_batch), segmentPartCounter) in enumerate(train_data):
            cur_state, _, batch_error, batch_acc = sess.run(
                [self.outputs, self.train_step, self.error, self.accuracy],
                feed_dict=merge_dicts({self.inputs: input_batch,
                                       self.targets: target_batch,
                                       self.init_state: cur_state}, extraFeedDict)
            )

            if (segmentPartCounter + 1) % train_data.segment_part_count == 0:
                cur_state = zeroState()
                # only care about the last output of the training error and acc of the rnn
                # so include it in if-statement
                train_error += batch_error
                train_accuracy += batch_acc
                # train_error.append(batch_error)
                # train_accuracy.append(batch_acc)

        train_error /= num_batches
        train_accuracy /= num_batches
        # assert len(train_error) == num_batches
        # assert len(train_accuracy) == num_batches
        # train_error = np.mean(train_error)
        # train_accuracy = np.mean(train_accuracy)

        return train_error, train_accuracy

    def getGraph(self, num_steps, state_size, learningRate=__default_learning_rate, verbose=True):
        if verbose:
            print "learning rate: {}".format(learningRate)

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(self.dtype, [self.batch_size, num_steps, self.segment_len],
                                        name='input_placeholder')

                targets = tf.placeholder(self.dtype, [self.batch_size, self.num_classes],
                                         name='labels_placeholder')

                init_state = tf.placeholder(self.dtype, [self.batch_size, state_size],
                                            name='previous_state_placeholder')

            # list where each item have dim 50 x 25
            rnn_inputs = tf.unpack(inputs, axis=1, name='rnn_inputs')

            def getRnn_W():
                input_dim, output_dim = self.segment_len + state_size, state_size
                return tf.get_variable(
                    'W', [input_dim, output_dim],
                    initializer=tf.truncated_normal_initializer(
                        stddev=2. / (input_dim + output_dim) ** 0.5
                    )
                )

            def getRnn_b():
                return tf.get_variable('b', [state_size],
                                       initializer=tf.constant_initializer(0.0))

            with tf.variable_scope('rnn_cell'):
                W = getRnn_W()
                b = getRnn_b()

            def rnn_cell(rnn_input, state):
                with tf.variable_scope('rnn_cell', reuse=True):
                    W = getRnn_W()
                    b = getRnn_b()

                    rnn_cell_out = tf.tanh(tf.matmul(
                        tf.concat(1, [rnn_input, state]), W
                        # concat dimension, inputs, so you see that both the state and the inputs are being treated as one
                    ) + b)

                return rnn_cell_out

            state = init_state
            rnn_outputs = []
            for rnn_input in rnn_inputs:
                state = rnn_cell(rnn_input, state)
                rnn_outputs.append(state)

            # as we see here the outputs are the state outputs of each rnn.

            outputs = rnn_outputs[-1]  # final state

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

                logits = tf.matmul(outputs, W) + b

            with tf.name_scope('error'):
                error = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, targets)
                )

            with tf.name_scope('softmax'):  # this is only for kaggle
                softmax = tf.nn.softmax(logits)

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),
                                                           tf.argmax(targets, 1)), dtype=self.dtype))

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

            init = tf.global_variables_initializer()

        self.init = init
        self.outputs = outputs
        self.inputs = inputs
        self.targets = targets
        self.init_state = init_state
        self.train_step = train_step
        self.error = error
        self.accuracy = accuracy
        self.logits = logits
        self.softmax = softmax

        return graph
