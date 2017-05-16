from __future__ import division

import sys

mlpdir = '/home/studenthp/pligor.george@gmail.com/' + \
         'msc_Artificial_Intelligence/mlp_Machine_Learning_Practical/mlpractical'
sys.path.append(mlpdir)
import os
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider
from mlp.data_providers import DataProvider
import tensorflow as tf
from mylibs.tf_helper import tfMSE, validateEpoch
from mylibs.batch_norm import fully_connected_layer_with_batch_norm_and_l2
from mylibs.jupyter_notebook_helper import getRunTime, initStats, gatherStats
from mylibs.py_helper import merge_dicts


# num_instances = 9950 #used to be when we had testing
class MSD10Genre_Teacher_DataProvider(DataProvider):  # MSD10GenreDataProvider
    """Data provider for Million Song Dataset 10-genre classification task."""

    num_classes = 10

    def __init__(self, dataset_filename, logits_filename, batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None,
                 verbose=True):  # here shuffling is OK because targets were created sequentially
        # check a valid which_set was provided

        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(os.environ['MLP_DATA_DIR'], dataset_filename)
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs = loaded['inputs']

        # flatten inputs to vectors and upcast to float32
        inputs = inputs.reshape((inputs.shape[0], -1)).astype('float32')

        # label map gives strings corresponding to integer label targets
        self.label_map = loaded['label_map']
        # pass the loaded data to the parent class __init__

        targets = np.load(logits_filename)

        if verbose:
            print "the inputs have {} length and the targets have {} length".format(len(inputs), len(targets))
        # inputs = inputs[:len(targets)] # trim inputs
        assert len(inputs) == len(targets)

        super(MSD10Genre_Teacher_DataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


class StudentNN(object):
    def __init__(self, batch_size, rng, dtype, config):
        super(StudentNN, self).__init__()
        self.outputs = None
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
        self.training = None
        self.keep_prob_input = None
        self.keep_prob_hidden = None
        self.train_step = None

    def teach_student(self, hidden_dim, lamda2, learning_rate, epochs, input_keep_prob, hidden_keep_prob,
                      dataset_filename, logits_filename):
        graph = self.loadAndGetGraph(hidden_dim=hidden_dim, lamda2=lamda2, learningRate=learning_rate)

        return self.trainAndValidate(
            dataset_filename=dataset_filename,
            logits_filename=logits_filename,
            graph=graph,
            epochs=epochs,
            input_keep_prob=input_keep_prob,
            hidden_keep_prob=hidden_keep_prob
        )

    def validate(self, data_provider, graph, epochs=35, verbose=True):
        if verbose:
            print "epochs: %d" % epochs

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            stats, keys = initStats(epochs)
            runTimes = []

            for e in range(epochs):
                (valid_error, valid_accuracy), runTime = getRunTime(lambda: validateEpoch(
                    inputs=self.inputs,
                    targets=self.targets,
                    sess=sess,
                    valid_data=data_provider,
                    error=self.error,
                    accuracy=self.accuracy,
                    extraFeedDict={self.training: False},
                    keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                ))

                runTimes.append(runTime)

                if verbose:
                    print 'End epoch %02d (%.3f secs): err(valid)=%.2f, acc(valid)=%.2f, ' % \
                          (e + 1, runTime, valid_error, valid_accuracy)

                stats = gatherStats(e, 0., 0., valid_error, valid_accuracy, stats)
        if verbose:
            print

        return stats, keys, runTimes

    def trainAndValidate(self, dataset_filename, logits_filename,
                         graph, input_keep_prob, hidden_keep_prob, epochs=35, verbose=True):
        if verbose:
            print "epochs: %d" % epochs
            print "input_keep_prob: %f" % input_keep_prob
            print "hidden_keep_prob: %f" % hidden_keep_prob

        train_data, valid_data = self.loadAndGetDataProviders(dataset_filename=dataset_filename,
                                                              logits_filename=logits_filename)

        with tf.Session(graph=graph, config=self.config) as sess:
            sess.run(self.init)

            stats, keys = initStats(epochs)

            for e in range(epochs):
                (train_error, train_accuracy), runTime = getRunTime(
                    lambda:
                    self.trainEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        train_data=train_data,
                        train_step=self.train_step,
                        error=self.error,
                        accuracy=self.accuracy,
                        extraFeedDict={
                            self.keep_prob_input: input_keep_prob,
                            self.keep_prob_hidden: hidden_keep_prob,
                            self.training: True
                        })
                )

                # print 'End epoch %02d (%.3f secs): err(train)=%.2f acc(train)=%.2f' % (e+1, runTime, train_error,train_accuracy)

                if (e + 1) % 1 == 0:
                    valid_error, valid_accuracy = validateEpoch(
                        inputs=self.inputs,
                        targets=self.targets,
                        sess=sess,
                        valid_data=valid_data,
                        error=self.error,
                        accuracy=self.accuracy,
                        extraFeedDict={self.training: False},
                        keep_prob_keys=[self.keep_prob_input, self.keep_prob_hidden]
                    )

                    # print((' ' * 27) + 'err(valid)={0:.2f} acc(valid)={1:.2f}'.format(valid_error, valid_accuracy))

                if verbose:
                    print 'End epoch %02d (%.3f secs):err(train)=%.2f,acc(train)=%.2f,err(valid)=%.2f,acc(valid)=%.2f, ' % \
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
            # curOutputs, \
            _, batch_error, batch_acc = sess.run(
                [
                    # self.outputs,
                    train_step, error, accuracy,
                ],
                feed_dict=merge_dicts({inputs: input_batch, targets: target_batch}, extraFeedDict)
            )

            train_error += batch_error
            train_accuracy += batch_acc

            # print curOutputs[:5, :3]

        train_error /= num_batches

        train_accuracy /= num_batches

        return train_error, train_accuracy

    # ((50, 3000), (50, 10))
    def loadAndGetDataProviders(self, dataset_filename, logits_filename):
        self.train_data = MSD10Genre_Teacher_DataProvider(
            dataset_filename=dataset_filename, logits_filename=logits_filename,
            batch_size=self.batch_size, rng=self.rng
        )
        self.valid_data = MSD10GenreDataProvider('test', batch_size=self.batch_size, rng=self.rng)

        return self.train_data, self.valid_data

    # tf.reset_default_graph() #kind of redundant statement
    def loadAndGetGraph(self,
                        learningRate=1e-4,  # default of Adam is 1e-3
                        lamda2=1e-2,
                        inputDim=3000,
                        numClasses=10,
                        hidden_dim=1000,
                        verbose=True):
        # momentum = 0.5

        if verbose:
            print "lamda2: %f" % lamda2
            print "hidden dim: %d" % hidden_dim
            print "learning rate: %f" % learningRate

        graph = tf.Graph()  # create new graph

        with graph.as_default():
            with tf.name_scope('data'):
                inputs = tf.placeholder(self.dtype, [None, inputDim], 'inputs')
                targets = tf.placeholder(self.dtype, [None, numClasses], 'targets')

            with tf.name_scope('params'):
                training = tf.placeholder(tf.bool, name="training")

            with tf.name_scope("dropout_inputs"):
                keep_prob_input = tf.placeholder(self.dtype)
                inputs_prob = tf.nn.dropout(inputs, keep_prob_input)

            with tf.name_scope('fully_connected'):
                hiddenLayer, hiddenRegularizer = fully_connected_layer_with_batch_norm_and_l2(
                    0, inputs_prob,
                    inputDim, hidden_dim,
                    nonlinearity=tf.nn.tanh,
                    training=training,
                    lamda2=lamda2
                )

            with tf.name_scope("dropout_hidden"):
                keep_prob_hidden = tf.placeholder(self.dtype)
                hidden_layer_prob = tf.nn.dropout(hiddenLayer, keep_prob_hidden)

            with tf.name_scope('readout_output_layer'):
                outputs, readoutRegularizer = fully_connected_layer_with_batch_norm_and_l2(
                    1, hidden_layer_prob,
                    hidden_dim, numClasses,
                    training=training,
                    nonlinearity=tf.identity,
                    lamda2=lamda2
                )

            with tf.name_scope('error'):
                def train_error():
                    return tfMSE(outputs, targets)

                def valid_error():
                    return tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)
                    )

                error = tf.cond(training, train_error, valid_error) + hiddenRegularizer + readoutRegularizer

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs, 1),
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
        self.training = training
        self.keep_prob_input = keep_prob_input
        self.keep_prob_hidden = keep_prob_hidden
        self.train_step = train_step
        self.outputs = outputs

        return graph
