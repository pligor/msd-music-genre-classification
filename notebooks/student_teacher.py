
# coding: utf-8

# ### Imports, Inits, and helper functions

# In[1]:

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

tensorboardLogdir = 'tf_cw{}_{}'.format(coursework, part)

curDtype = tf.float32

reluBias = 0.1

batch_size = 50
numTestSongs = 9950
numClasses = 10


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


# ### MSD 10 genre task

# In[6]:

#we want a provider that has as inputs our regular inputs but as outputs the outputs
#we got from the teacher deep-state-of-the-art neural net


# In[7]:

from models.teacher_student_nn import MSD10Genre_Teacher_DataProvider, StudentNN


# In[8]:

#studentNN = StudentNN(batch_size=batch_size,rng=rng,dtype=curDtype, config=config)


# In[9]:

#show_graph(studentNN.loadAndGetGraph())
dataset_filename = 'msd-10-genre-train_valid.npz'


# In[10]:

logits_filename = 'rnn_logits.npy'


# In[11]:

os.path.isfile(logits_filename)


# ## Bayesian Opt

# In[12]:

res_gp_save_filename = 'student_teacher_res_gp.pickle'
statsCollectionFilename = 'student_teacher_bay_opt_statsCollection.npy'
best_params_student_teacher_filename = 'student_teacher_best_params.npy'


# In[13]:

def objective(params): # Here we define the metric we want to minimise
    input_keep_prob, hidden_keep_prob, hidden_dim, lamda2 = params
    
    epochs = 20
    learning_rate = 1e-4
        
    studentNN = StudentNN(batch_size=batch_size, rng=rng, dtype=curDtype, config=config)

    stats, keys = studentNN.teach_student(
        hidden_dim = hidden_dim,
        lamda2 = lamda2,
        learning_rate = learning_rate,
        epochs = epochs,
        input_keep_prob = input_keep_prob,
        hidden_keep_prob = hidden_keep_prob,
        dataset_filename = dataset_filename,
        logits_filename = logits_filename,
    )
    
    #save everytime in case it crashes
    filename = statsCollectionFilename
    statsCollection = np.load(filename)[()] if os.path.isfile(filename) else dict()
    statsCollection[tuple(params)] = stats
    np.save(filename, statsCollection)
    
    if plotting:
        fig_1, ax_1, fig_2, ax_2 = plotStats(stats, keys)
        plt.show()
    
    validAccs = stats[:, -1]
    length10percent = len(validAccs) // 10
    best10percent = np.sort(validAccs)[-length10percent:]
    # We want to maximise the MEAN validation accuracy,
    # i.e. minimise minus
    return -np.mean(best10percent)


# In[14]:

inputKeepProbSpace = Real(0.5, 1.0, "uniform")
hiddenKeepProbSpace = Real(0.5, 1.0, "uniform")
hiddenDimSpace = Integer(20, 2000)
lamda2Space = Real(1e-3, 10, "log-uniform")
space  = [inputKeepProbSpace, hiddenKeepProbSpace, hiddenDimSpace, lamda2Space]


# TARGET IS 58% as the original Deep Neural Net

# In[15]:

if jupyterNotebookEnabled:
    get_ipython().magic(u'%time')

#this might crash so you need to run it outside as a python file (file -> save as python)
if not os.path.isfile(res_gp_save_filename):
    if os.path.isfile(statsCollectionFilename):
        os.remove(statsCollectionFilename) #erase before executing
    
    res_gp = gp_minimize(
        func=objective, # function that we wish to minimise
        dimensions=space, #the search space for the hyper-parameters
        #x0=x0, #inital values for the hyper-parameters
        n_calls=25, #number of times the function will be evaluated
        random_state = seed, #random seed
        n_random_starts=5,
            #before we start modelling the optimised function with a GP Regression
            #model, we want to try a few random choices for the hyper-parameters.
        kappa=1.9 #trade-off between exploration vs. exploitation.
    )


# In[ ]:

res_gp = loadPythonVarOrSave(res_gp_save_filename, lambda : res_gp)


# In[ ]:

if os.path.isfile(best_params_student_teacher_filename):
    best_params_student_teacher = np.load(best_params_student_teacher_filename)
else:
    np.save(best_params_student_teacher_filename, res_gp.x)
    best_params_student_teacher = res_gp.x
    
best_params_student_teacher


# In[ ]:

print "Best score with Bayesian optimisation: {:.3f}".format(-res_gp.fun)
print
print "Best parameters with Bayesian optimisation:"
print res_gp.x

