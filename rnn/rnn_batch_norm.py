from __future__ import division
import tensorflow as tf
import numpy as np
import os
from manual_rnn import ManualRNN
from mylibs.jupyter_notebook_helper import getRunTime, initStats, gatherStats, DynStats
from mylibs.batch_norm import fully_connected_layer_with_batch_norm

class RNNBatchNorm(ManualRNN):  # CrossValidator, RnnModelInterface, LogitsGatherer, KaggleRNN
    def __init__(self, batch_size, rng, dtype, config, segment_count, segment_len, seed=16011984, num_classes=10):
        super(RNNBatchNorm, self).__init__(batch_size=batch_size, rng=rng, dtype=dtype, config=config,
                                           segment_count=segment_count, segment_len=segment_len, seed=seed,
                                           num_classes=num_classes)

        self.training = None

    def run_until(self, targetValidAcc, maxEpochs, learning_rate, num_steps, state_size, verbose=True):
        """onLogits(sess, logits, create_feed_dict, filename = 'msd-25-submission.csv')"""

        train_data, valid_data = self.loadAndGetDataProviders(num_steps=num_steps)

        graph = self.getGraph(num_steps=num_steps, state_size=state_size, learningRate=learning_rate)

        if verbose:
            print "rnn steps: %d" % train_data.num_steps
            print "state size: %d" % state_size

        max_valid_acc = 0.

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            dynStats = DynStats()

            valid_accuracy = 0.
            epochCounter = 0

            while valid_accuracy < targetValidAcc and epochCounter < maxEpochs:
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda: self.trainEpoch(state_size=state_size, sess=sess, train_data=train_data, extraFeedDict={
                        self.training: True
                    })
                )

                # if (e + 1) % 1 == 0: #better calc it always
                valid_error, valid_accuracy = self.validateEpoch(
                    state_size=state_size, sess=sess, valid_data=valid_data, extraFeedDict={
                        self.training: False
                    }
                )

                if valid_accuracy > max_valid_acc:
                    max_valid_acc = valid_accuracy

                if verbose:
                    print 'End epoch %02d (%.3f secs): err(train)=%.2f, acc(train)=%.2f, err(valid)=%.2f, acc(valid)=%.2f, ' % \
                          (epochCounter + 1, runTime, train_error, train_accuracy, valid_error, valid_accuracy)
                    print

                dynStats.gatherStats(train_error=train_error, train_accuracy=train_accuracy,
                                     valid_error=valid_error, valid_accuracy=valid_accuracy)

                epochCounter += 1

            metric = epochCounter if epochCounter < maxEpochs else (maxEpochs * targetValidAcc / max_valid_acc)

        return dynStats.stats, metric

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
                    lambda: self.trainEpoch(state_size=state_size, sess=sess, train_data=train_data, extraFeedDict={
                        self.training: True
                    })
                )

                # if (e + 1) % 1 == 0: #better calc it always
                valid_error, valid_accuracy = self.validateEpoch(
                    state_size=state_size, sess=sess, valid_data=valid_data, extraFeedDict={
                        self.training: False
                    }
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

    def getRnn_W(self, state_size):
        input_dim, output_dim = self.segment_len + state_size, state_size
        return tf.get_variable(
            'W', [input_dim, output_dim],
            initializer=tf.truncated_normal_initializer(
                stddev=2. / (input_dim + output_dim) ** 0.5
            )
        )

    @staticmethod
    def getRnn_b(state_size):
        return tf.get_variable('b', [state_size],
                               initializer=tf.constant_initializer(0.0))

    @staticmethod
    def get_pop_mean(outputDim):
        return tf.get_variable(name="pm", shape=[outputDim], initializer=tf.constant_initializer(0.), trainable=False)

    @staticmethod
    def get_pop_var(outputDim):
        return tf.get_variable(name="pv", shape=[outputDim], initializer=tf.constant_initializer(1.), trainable=False)

    @staticmethod
    def get_beta_offset(outputDim):
        return tf.get_variable(name="bo", shape=[outputDim], initializer=tf.constant_initializer(0.))

    @staticmethod
    def get_scale_gamma(outputDim):
        return tf.get_variable(name="sg", shape=[outputDim], initializer=tf.constant_initializer(1.))

    def getGraph(self, num_steps, state_size, learningRate=1e-4):
        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(self.dtype, [self.batch_size, num_steps, self.segment_len],
                                        name='input_placeholder')

                targets = tf.placeholder(self.dtype, [self.batch_size, self.num_classes],
                                         name='labels_placeholder')

                init_state = tf.placeholder(self.dtype, [self.batch_size, state_size],
                                            name='previous_state_placeholder')

            with tf.name_scope('params'):
                training = tf.placeholder(tf.bool, name="training")

            # list where each item have dim 50 x 25
            rnn_inputs = tf.unpack(inputs, axis=1, name='rnn_inputs')

            with tf.variable_scope('rnn_cell'):
                _ = self.getRnn_W(state_size=state_size)
                _ = self.getRnn_b(state_size=state_size)
                _ = self.get_pop_mean(outputDim=state_size)
                _ = self.get_pop_var(outputDim=state_size)
                _ = self.get_beta_offset(outputDim=state_size)
                _ = self.get_scale_gamma(outputDim=state_size)

            def rnn_cell(rnn_input, the_state):
                with tf.variable_scope('rnn_cell', reuse=True):
                    with tf.name_scope('rnn_cell_affine_layer'):
                        W = self.getRnn_W(state_size=state_size)
                        b = self.getRnn_b(state_size=state_size)

                        out_affine = tf.matmul(
                            tf.concat(1, [rnn_input, the_state]), W
                            # concat dimension, inputs, so you see that both the state and the inputs are being treated as one
                        ) + b

                    with tf.name_scope('rnn_cell_batch_norm'):
                        batchNorm = self.batchNormWrapper_byExponentialMovingAvg(
                            out_affine, training,
                            get_pop_mean=self.get_pop_mean,
                            get_pop_var=self.get_pop_var,
                            get_beta_offset=self.get_beta_offset,
                            get_scale_gamma=self.get_scale_gamma)

                    with tf.name_scope('rnn_cell_act_func'):
                        rnn_cell_out = tf.tanh(batchNorm)

                return rnn_cell_out

            state = init_state
            rnn_outputs = []
            for rnn_inpt in rnn_inputs:
                state = rnn_cell(rnn_inpt, state)
                rnn_outputs.append(state)

            # as we see here the outputs are the state outputs of each rnn.

            final_state_rnn_outputs = rnn_outputs[-1]  # final state

            with tf.variable_scope('readout'):
                # readout_weights = tf.Variable(
                #     tf.truncated_normal(
                #         [input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5
                #     ),
                #     name='readout_weights'
                # )
                # readout_biases = tf.Variable(tf.zeros([output_dim]),
                #                              name='readout_biases')

                logits = fully_connected_layer_with_batch_norm(
                    "readout",
                    final_state_rnn_outputs,
                    input_dim = state_size,
                    output_dim = self.num_classes,
                    nonlinearity=tf.identity,
                    training=training,
                )

                #logits = tf.matmul(final_state_rnn_outputs, readout_weights) + readout_biases

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
        self.outputs = final_state_rnn_outputs
        self.inputs = inputs
        self.targets = targets
        self.init_state = init_state
        self.train_step = train_step
        self.error = error
        self.accuracy = accuracy
        self.logits = logits
        self.softmax = softmax
        self.training = training

        return graph

    @staticmethod
    def batchNormWrapper_byExponentialMovingAvg(ins, training, epsilon=1e-3,
                                                get_pop_mean=None, get_pop_var=None,
                                                get_beta_offset=None, get_scale_gamma=None):
        outputDim = ins.get_shape()[-1]

        pop_mean = tf.Variable(tf.zeros(outputDim), trainable=False,
                               name='pm') if get_pop_mean is None else get_pop_mean(outputDim)
        pop_var = tf.Variable(tf.ones(outputDim), trainable=False, name='pv') if get_pop_var is None else get_pop_var(
            outputDim)

        beta_offset = tf.Variable(tf.zeros(outputDim), name='bo') if get_beta_offset is None else get_beta_offset(
            outputDim)
        scale_gamma = tf.Variable(tf.ones(outputDim), name='sg') if get_scale_gamma is None else get_scale_gamma(
            outputDim)

        # given that on axis=0 is where the batches extend (we want mean and var for each attribute)
        batch_mean, batch_var = tf.nn.moments(ins, [0])

        decay = 0.999  # use numbers closer to 1 if you have more data
        mean_of_train = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))  # we just want to use the
        var_of_train = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([mean_of_train, var_of_train]):
            normalized = tf.nn.batch_normalization(ins,
                                                   tf.cond(training, lambda: batch_mean, lambda: pop_mean),
                                                   tf.cond(training, lambda: batch_var, lambda: pop_var),
                                                   beta_offset, scale_gamma, epsilon)

        return normalized
