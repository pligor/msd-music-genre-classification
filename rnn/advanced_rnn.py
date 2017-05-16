import tensorflow as tf
import numpy as np

from mylibs.jupyter_notebook_helper import getRunTime, initStats, gatherStats
from mylibs.tf_helper import validateEpoch
from mylibs.py_helper import merge_dicts
from my_basic_rnn import MyBasicRNN


class AdvancedRNN(MyBasicRNN):
    num_classes = 10

    def __init__(self, batch_size, rng, dtype, config, segment_count, segment_len):
        super(AdvancedRNN, self).__init__(batch_size, rng, dtype, config, segment_count, segment_len)

    def rnn_native_tf(self, state_size, num_steps, epochs,
                      num_of_last_to_combine=1, stride=1, verbose=True):
        if verbose:
            print "number of last rnn outputs to combine: {}".format(num_of_last_to_combine)
            print "stride: {}".format(stride)

        train_data, valid_data = self.loadAndGetDataProviders(num_steps=num_steps)

        graph = self.loadAndGetGraph(num_steps=num_steps, state_size=state_size,
                                     numOfLastToCombine=num_of_last_to_combine, stride=stride)

        stats, keys = self.trainAndValidate(
            train_data, valid_data, state_size, graph, num_steps=num_steps, epochs=epochs, verbose=verbose
        )

        return stats, keys

    def loadAndGetGraph(self, num_steps, state_size,
                        learningRate=1e-4,  # default of Adam is 1e-3
                        numOfLastToCombine=1, stride=1, verbose=True):
        """numOfLastToCombine is meant to take into account the last 3 or 4 or 5 or 120 segments even if stride reduces them
        stride is useful for picking only a few items to take into account instead of everything"""
        assert 1 <= numOfLastToCombine <= self.segment_count
        assert stride >= 1  # even if it is very last it is going to take into account the last one in any case

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
            )  # each rnn_output from rnn_outputs has 50 x state size

            with tf.variable_scope('rnn_outputs_multiplex'):
                rnn_outputs_packed = tf.pack(rnn_outputs, axis=1)  # 50 x 120 x state size

                # aa = np.arange(10)
                # aa[-1:-(5+1):-2] #five last with a step of two
                # setting it with python code break the flow
                # rnn_outputs_of_interest = rnn_outputs[-1:-(numOfLastToCombine + 1):-stride]

                reversed_rnn_outputs = tf.reverse_v2(rnn_outputs_packed, axis=[1])

                rnn_outputs_of_interest = tf.strided_slice(
                    reversed_rnn_outputs,
                    [0, 0, 0],
                    [int(reversed_rnn_outputs.get_shape()[0]), numOfLastToCombine, int(reversed_rnn_outputs.get_shape()[2])],
                    [1, stride, 1]
                )

                # rnn_outputs_of_interest_packed = tf.pack(rnn_outputs_of_interest, axis=1)

                # rnn_outputs_of_interest_combined = final_state
                # rnn_outputs_of_interest_combined = tf.concat(concat_dim=1, values=rnn_outputs_of_interest)

                rnn_outputs_of_interest_combined = tf.reshape(rnn_outputs_of_interest, (self.batch_size, -1))

                rnn_outputs_multiplex = tf.concat(concat_dim=1, values=[final_state, rnn_outputs_of_interest_combined])

            with tf.variable_scope('readout'):
                # input_dim = state_size * len(rnn_outputs_of_interest)
                input_dim = state_size * int(rnn_outputs_of_interest.get_shape()[1])
                assert rnn_outputs_of_interest_combined.get_shape()[-1] == input_dim

                input_dim += state_size  #because we are adding the final state

                if verbose:
                    if input_dim > 2000:
                        print "input dimensionality for readout layer is too large: {}".format(input_dim)

                output_dim = self.num_classes
                W = tf.Variable(
                    tf.truncated_normal(
                        [input_dim, output_dim], stddev=2. / (input_dim + output_dim) ** 0.5
                    ),
                    name='readout_weights'
                )
                b = tf.Variable(tf.zeros([output_dim]),
                                name='readout_biases')

                logits = tf.matmul(rnn_outputs_multiplex, W) + b  # shape: (50, 10)

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

        self.lastRnnOut = rnn_outputs[-1]
        self.finalState = final_state
        self.initialState = init_state
        self.firstRnnOut = rnn_outputs[0]
        self.allRnnOuts = rnn_outputs

        return graph


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
                    self.trainEpoch(
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

    def trainEpoch(self, inputs, targets, sess, train_data, train_step, error, accuracy, extraFeedDict={}):
        train_error = 0.
        train_accuracy = 0.

        num_batches = train_data.num_batches

        for step, (input_batch, target_batch) in enumerate(train_data):
            sess_out = sess.run(
                [self.firstRnnOut, self.initialState, self.lastRnnOut, self.finalState,
                 train_step, error, accuracy] + self.allRnnOuts,
                feed_dict=merge_dicts({inputs: input_batch, targets: target_batch}, extraFeedDict)
            )

            first_rnn_out, cur_initial_state, last_rnn_out, cur_fin_state, _, batch_error, batch_acc = sess_out[:7]

            all_rnn_outs = sess_out[7:]

            if step % 100 == 0:
                one_like_initial_state = False
                one_like_final_state = False
                for counter, rnn_out in enumerate(all_rnn_outs):
                    cur_check = np.allclose(cur_initial_state, rnn_out)
                    one_like_initial_state = one_like_initial_state or cur_check
                    if cur_check:
                        print "for initial state counter: {}".format(counter)

                for counter, rnn_out in enumerate(all_rnn_outs):
                    cur_check = np.allclose(cur_fin_state, rnn_out)
                    one_like_final_state = one_like_final_state or cur_check
                    if cur_check:
                        print "for final state counter: {}".format(counter)


                #assert np.allclose(last_rnn_out, cur_fin_state) # fails
                #assert np.allclose(first_rnn_out, cur_initial_state) #fails as well
                print "one_like_initial_state"
                print one_like_initial_state
                print "one_like_final_state"
                print one_like_final_state

            train_error += batch_error
            train_accuracy += batch_acc

        train_error /= num_batches

        train_accuracy /= num_batches

        return train_error, train_accuracy
