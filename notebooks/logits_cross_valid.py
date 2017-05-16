
# coding: utf-8

# ### Imports, Inits, and helper functions

# In[16]:

jupyterNotebookEnabled = False
plotting = False

if jupyterNotebookEnabled:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')


# In[2]:

import sys
import os
mlpdir = os.path.expanduser(
    '~/pligor.george@gmail.com/msc_Artificial_Intelligence/mlp_Machine_Learning_Practical/mlpractical'
)
sys.path.append(mlpdir)


# In[3]:

import pickle
import skopt
from skopt.plots import plot_convergence
import datetime
import time
import tensorflow as tf
import numpy as np
    
import matplotlib.pyplot as plt

from mylibs.tf_helper import tfRMSE, tfMSE, fully_connected_layer, trainEpoch, validateEpoch

from mylibs.py_helper import merge_dicts
    
from mylibs.dropout_helper import constructProbs

from skopt.space.space import Real, Integer
from skopt import gp_minimize

from mylibs.jupyter_notebook_helper import show_graph

from mlp.data_providers import DataProvider,     MSD10GenreDataProvider, MSD25GenreDataProvider,    MSD10Genre_Autoencoder_DataProvider, MSD10Genre_StackedAutoEncoderDataProvider
    
from mylibs.batch_norm import batchNormWrapper_byExponentialMovingAvg, fully_connected_layer_with_batch_norm

from mylibs.batch_norm import fully_connected_layer_with_batch_norm_and_l2

from mylibs.stacked_autoencoder_pretrainer import     constructModelFromPretrainedByAutoEncoderStack,    buildGraphOfStackedAutoencoder, executeNonLinearAutoencoder
    
from mylibs.jupyter_notebook_helper import getRunTime, getTrainWriter, getValidWriter,    plotStats, initStats, gatherStats
    
from rnn.manual_rnn import ManualRNN
from collections import OrderedDict


# In[4]:

if jupyterNotebookEnabled:
    get_ipython().magic(u'matplotlib inline')
    
coursework, part = 4, 6
saving = False

seed = 16011984
rng = np.random.RandomState(seed=seed)

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

figcount = 0

tensorboardLogdir = 'tf_cw%d_%d' % (coursework, part)

curDtype = tf.float32

reluBias = 0.1

batch_size = 50
#numTestSongs = 9950
numClasses = 10

segment_count = 120
segment_len = 25


# In[5]:

#http://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
def loadPythonVarOrSave(filename, lambda_var):
    if os.path.isfile(filename):
        with open(filename) as f:  # Python 3: open(..., 'rb')
            (thevar, ) = pickle.load(f)
    else:
        thevar = lambda_var()
        with open(filename, 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([thevar], f)
            
    return thevar


# In[6]:

best_params_filename = 'best_params_rnn.npy'


# In[7]:

from mylibs.jupyter_notebook_helper import DynStats


# ### MSD 10 genre task

# In[8]:

best_params = np.load(best_params_filename)
state_size, num_steps = best_params
best_params


# In[9]:

manualRNN = ManualRNN(batch_size=batch_size,rng=rng, dtype=curDtype, config=config,
                      segment_count = segment_count, segment_len=segment_len, seed=seed)


# In[10]:

def onTrainEnd(stats, logits_dict):
    if plotting:
        fig_1, ax_1, fig_2, ax_2 = plotStats(stats, DynStats.keys)
        plt.show()


# In[11]:

stats_list, logits_arr = manualRNN.cross_gather_logits(
    n_splits=10,
    epochs=30,
    state_size = state_size,
    num_steps=num_steps,
    onTrainEnd=onTrainEnd
)


# In[13]:

np.save('rnn_logits.npy', logits_arr)
np.save('rnn_stats_list_cross_validate.npy', stats_list)

print "FINITTO la musica"
