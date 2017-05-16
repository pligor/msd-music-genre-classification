# -*- coding: utf-8 -*-
"""Jupyter Notebook Helpers"""

from IPython.display import display, HTML
import datetime
# from time import time
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from collections import OrderedDict
import operator


def show_graph(graph_def, frame_size=(900, 600)):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(height=frame_size[1], data=repr(str(graph_def)), id='graph' + timestamp)
    iframe = """
        <iframe seamless style="width:{width}px;height:{height}px;border:0" srcdoc="{src}"></iframe>
    """.format(width=frame_size[0], height=frame_size[1] + 20, src=code.replace('"', '&quot;'))
    display(HTML(iframe))


def getRunTime(function):  # a = lambda _ = None : 3 or #a = lambda : 3
    run_start_time = time.time()
    result = function()
    run_time = time.time() - run_start_time
    return result, run_time


def getWriter(key, graph, folder):
    # tensorboard --logdir=<folder>

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return tf.summary.FileWriter(
        logdir=os.path.join(folder, timestamp, key),
        graph=graph
    )


def getTrainWriter():
    return getWriter('train', tf.get_default_graph())


def getValidWriter():
    return getWriter('valid', tf.get_default_graph())


def plotStats(stats, keys, stats_interval=1):
    # Plot the change in the validation and training set error over training.

    # stats[0:, keys[k]] #0 epoch number
    # stats[1:, keys[k]] #1 for training and validation
    # keys is from string to index
    # stats shape is [epochs, 4]

    fig_1 = plt.figure(figsize=(12, 6))
    ax_1 = fig_1.add_subplot(111)

    ax_1.hold(True)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_1.hold(False)

    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(12, 6))
    ax_2 = fig_2.add_subplot(111)

    ax_2.hold(True)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys[k]], label=k)
    ax_2.hold(False)

    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    # plt.show() better do it outside when you want it
    return fig_1, ax_1, fig_2, ax_2


def initStats(epochs):
    stats = np.zeros((epochs, 4))

    keys = {
        'error(train)': 0,
        'acc(train)': 1,
        'error(valid)': 2,
        'acc(valid)': 3
    }

    return stats, keys


def gatherStats(e, train_error, train_accuracy, valid_error, valid_accuracy, stats):
    stats[e, 0] = train_error
    stats[e, 1] = train_accuracy
    stats[e, 2] = valid_error
    stats[e, 3] = valid_accuracy

    return stats


class DynStats(object):
    def __init__(self):
        super(DynStats, self).__init__()
        self.__stats = []

    keys = {
        'error(train)': 0,
        'acc(train)': 1,
        'error(valid)': 2,
        'acc(valid)': 3
    }

    def gatherStats(self, train_error, train_accuracy, valid_error, valid_accuracy):
        self.__stats.append(
            np.array([train_error, train_accuracy, valid_error, valid_accuracy])  # KEEP THE ORDER
        )

        return self.stats

    @property
    def stats(self):
        return np.array(self.__stats)


def renderStatsCollection(statsCollection, epochs, label_texts, title='Training Error', k='error(train)'):
    fig = plt.figure(figsize=(12, 6))

    maxValidAccs = OrderedDict([(key, max(val[:, -1])) for key, val in statsCollection.iteritems()])
    highValidAccs = sorted(maxValidAccs.items(), key=operator.itemgetter(1))[::-1]
    highValidAccs = OrderedDict(
        highValidAccs[:7])  # only seven because these are the colors support by default by matplotlib

    for key in statsCollection:
        label = ", ".join(
            [(label_texts[i] + ": " + str(val)) for i, val in enumerate(key)]
        )
        stats = statsCollection[key]
        keys = DynStats.keys
        xValues = np.arange(1, stats.shape[0])
        yValues = stats[1:, keys[k]]

        if key in highValidAccs.keys():
            plt.plot(xValues, yValues, label=label)
        else:
            plt.plot(xValues, yValues, c='lightgrey')
        plt.hold(True)

    plt.hold(False)
    plt.legend(loc=0)
    plt.title(title + ' over {} epochs'.format(epochs))
    plt.xlabel('Epoch number')
    plt.ylabel(title)
    plt.grid()
    plt.show()

    return fig  # fig.savefig('cw%d_part%d_%02d_fig.svg' % (coursework, part, figcount))
