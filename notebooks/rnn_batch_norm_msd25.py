
# coding: utf-8

# In[1]:

from __future__ import division


# # KAGGLE ONLY PURPOSES

# 2017
# 
# Machine Learning Practical
# 
# University of Edinburgh
# 
# Georgios Pligoropoulos - s1687568
# 
# Coursework 4 (part 8)

# ### Imports, Inits, and helper functions

# In[2]:

jupyterNotebookEnabled = False
plotting = False
coursework, part = 4, 8
saving = True

if jupyterNotebookEnabled:
    #%load_ext autoreload
    get_ipython().magic(u'reload_ext autoreload')
    get_ipython().magic(u'autoreload 2')


# In[3]:

import sys, os
mlpdir = os.path.expanduser(
    '~/pligor.george@gmail.com/msc_Artificial_Intelligence/mlp_Machine_Learning_Practical/mlpractical'
)
sys.path.append(mlpdir)


# In[4]:

from collections import OrderedDict
import skopt
from mylibs.jupyter_notebook_helper import show_graph
import datetime
import os
import time
import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider,    MSD10Genre_Autoencoder_DataProvider, MSD10Genre_StackedAutoEncoderDataProvider
import matplotlib.pyplot as plt
from mylibs.batch_norm import fully_connected_layer_with_batch_norm_and_l2
from mylibs.stacked_autoencoder_pretrainer import     constructModelFromPretrainedByAutoEncoderStack,    buildGraphOfStackedAutoencoder, executeNonLinearAutoencoder
    
from mylibs.jupyter_notebook_helper import getRunTime, getTrainWriter, getValidWriter,    plotStats, initStats, gatherStats, renderStatsCollection
    
from mylibs.tf_helper import tfRMSE, tfMSE, fully_connected_layer
    #trainEpoch, validateEpoch

from mylibs.py_helper import merge_dicts

from mylibs.dropout_helper import constructProbs

from mylibs.batch_norm import batchNormWrapper_byExponentialMovingAvg,    fully_connected_layer_with_batch_norm
    
import pickle
from skopt.plots import plot_convergence
from mylibs.jupyter_notebook_helper import DynStats
import operator
from skopt.space.space import Integer, Categorical, Real
from skopt import gp_minimize
from rnn.rnn_batch_norm import RNNBatchNorm

if jupyterNotebookEnabled:
    get_ipython().magic(u'matplotlib inline')


# In[5]:

seed = 16011984
rng = np.random.RandomState(seed=seed)

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

figcount = 0

tensorboardLogdir = 'tf_cw%d_%d' % (coursework, part)

curDtype = tf.float32

reluBias = 0.1


# In[6]:

batch_size = 50
segmentCount = 120
segmentLen = 25


# In[7]:

best_params_filename = 'rnn_msd25_best_params.npy'
stats_coll_filename = 'rnn_msd25_bay_opt_stats_coll.npy'
res_gp_save_filename = 'rnn_msd25_res_gp.pickle'
stats_final_run_filename = 'rnn_msd25_final_run_stats.npy'


# here the state size is equal to the number of classes because we have given to the last output all the responsibility.
# 
# We are going to follow a repetitive process. For example if num_steps=6 then we break the 120 segments into 20 parts
# 
# The output of each part will be the genre. We are comparing against the genre every little part 

# ### MSD 25 genre Bayesian Optimization

# In[8]:

num_classes=25


# In[9]:

numLens = np.sort(np.unique([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]))
assert np.all( segmentCount % numLens == 0 )
print len(numLens)
numLens


# In[10]:

rnnModel = RNNBatchNorm(batch_size=batch_size, rng=rng, dtype = curDtype, config=config,
                      segment_count=segmentCount, segment_len= segmentLen, num_classes=num_classes)


# In[11]:

#it cannot accept global variables for some unknown reason ...
def objective(params): # Here we define the metric we want to minimise    
    (state_size, num_steps, learning_rate) = params
    
    epochs = 20
    
    stats, keys = rnnModel.run_rnn(state_size = state_size, num_steps=num_steps, epochs = epochs, 
                                  learning_rate = learning_rate)
    
    #save everytime in case it crashes
    filename = stats_coll_filename
    statsCollection = np.load(filename)[()] if os.path.isfile(filename) else dict()
    statsCollection[(state_size, num_steps, learning_rate)] = stats
    np.save(filename, statsCollection)
    
    if plotting:
        fig_1, ax_1, fig_2, ax_2 = plotStats(stats, keys)
        plt.show()
    
    # We want to maximise validation accuracy, i.e. minimise minus validation accuracy
    validAccs = stats[:, -1]
    length10percent = max(len(validAccs) // 10, 1)
    best10percent = np.sort(validAccs)[-length10percent:]
    return -np.mean(best10percent)
    #return -max(stats[:, -1])


# In[12]:

#it cannot accept global variables for some unknown reason ...
def objective_min_epochs(params): # Here we define the metric we want to minimise    
    (state_size, num_steps, learning_rate) = params
    
    targetValidAcc = 0.23
    maxEpochs = 20
    
    stats, metric = rnnModel.run_until(targetValidAcc = targetValidAcc, maxEpochs=maxEpochs,
                                       learning_rate=learning_rate, num_steps=num_steps, state_size =state_size)

    
    #save everytime in case it crashes
    filename = stats_coll_filename
    statsCollection = np.load(filename)[()] if os.path.isfile(filename) else dict()
    statsCollection[(state_size, num_steps, learning_rate)] = stats
    np.save(filename, statsCollection)
    
    if plotting:
        fig_1, ax_1, fig_2, ax_2 = plotStats(stats, DynStats.keys)
        plt.show()
    
    # We want to minimize the amount of epochs required to reach 23% accuracy
    return metric


# In[13]:

stateSizeSpace = Integer(15, 600)
numStepSpace = Categorical(numLens)
learningRateSpace = Real(1e-6, 1e-1, prior="log-uniform")
space  = [stateSizeSpace, numStepSpace, learningRateSpace]


# In[14]:

if jupyterNotebookEnabled:
    get_ipython().magic(u'%time')

if not os.path.isfile(best_params_filename):
    if os.path.isfile(stats_coll_filename):
        os.remove(stats_coll_filename)
    
    res_gp = gp_minimize(
            func=objective_min_epochs, # function that we wish to minimise
            dimensions=space, #the search space for the hyper-parameters
            #x0=x0, #inital values for the hyper-parameters
            n_calls=50, #number of times the function will be evaluated
            random_state = seed, #random seed
            n_random_starts=5,
                #before we start modelling the optimised function with a GP Regression
                #model, we want to try a few random choices for the hyper-parameters.
            kappa=1.9 #trade-off between exploration vs. exploitation.
        )


# In[14]:

if os.path.isfile(best_params_filename):
    best_params = np.load(best_params_filename)
else:
    np.save(best_params_filename, res_gp.x)
    best_params = res_gp.x


# In[15]:

if os.path.isfile(res_gp_save_filename):
    with open(res_gp_save_filename) as f:  # Python 3: open(..., 'rb')
        (res_gp, ) = pickle.load(f)
else:
    with open(res_gp_save_filename, 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([res_gp], f)


# In[16]:

print best_params

