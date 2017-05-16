# -*- coding: utf-8 -*-
"""Stacked Autoencoder Pretrainer"""

import numpy as np
import tensorflow as tf
from tf_helper import tfMSE, fully_connected_layer
from jupyter_notebook_helper import getRunTime

def buildGraphOfStackedAutoencoder(inDim, variables, nonlinearity=tf.nn.tanh, avoidDeadNeurons = 0.,
                                  lastNonLinearity=tf.nn.tanh):
    graph = tf.Graph() #create new graph
    
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, inDim], 'inputs')

        #with tf.name_scope('fullyConnected'):
        
        layer = inputs
        #inDim = inDim
        
        for i, (weights, biases) in enumerate(variables): #[3000x400], [400,]
            outDim = len(biases)
            
            curNonLinearity = lastNonLinearity if i+1 == len(variables) else nonlinearity
            
            layer = fully_connected_layer(layer, inDim, outDim, nonlinearity = curNonLinearity, w=weights, b=biases,
                                         avoidDeadNeurons = avoidDeadNeurons)
            
            inDim = outDim

        init = tf.global_variables_initializer()
        
        outputs = layer
        
    return graph, init, inputs, outputs


def constructModelFromPretrainedByAutoEncoderStack(hiddenDimLayers, autoencoder, rng, dataProviderClass, config,
												   inputDim = 3000, batchSize=50):
    #preTrainedModels = []
    curInputDim = inputDim
    curProcessBatch = None
    
    variables = []
    
    #count = 0
    
    for hiddenDimLayer in hiddenDimLayers:
        dataProvider = dataProviderClass(
            'train', batch_size=batchSize, rng=rng, processBatch = curProcessBatch
        )
        
        _, weights, biases, _ = autoencoder(curInputDim, hiddenDimLayer, dataProvider, config=config) #graph, w, b, runTime
        
        variables.append((weights, biases))
        
        graph, init, inputs, outputs = buildGraphOfStackedAutoencoder(inputDim, variables) #recreate it from scratch actually
        
        #count += 1
        
        def processBatchFunc(inputBatch): #this is redefined automatically, no need to do any special code here
            """process batch should calculate fprop for inputBatch"""
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init)
            
                (outValue, ) = sess.run([outputs], feed_dict={inputs: inputBatch})
            
            #print "count now is: %d" % count

            return outValue
        
        curProcessBatch = processBatchFunc
        
        curInputDim = hiddenDimLayer
        
    return variables


def executeNonLinearAutoencoder(inputOutputDim, hiddenDim, dataProvider, config,
                                numEpochs=20, errorFunc = tfMSE,
                               nonLinearLayer1 = tf.nn.tanh, nonLinearLayer2=tf.nn.tanh, printing=True):
    #inputOutputDim = dataProvider.inputs.shape[1]
    
    graph = tf.Graph() #create new graph
    
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, [None, inputOutputDim], 'inputs')
        targets = tf.placeholder(tf.float32, [None, inputOutputDim], 'targets')

        with tf.name_scope('fullyConnected'):
            hidden_layer = fully_connected_layer(inputs, inputOutputDim, hiddenDim, nonlinearity = nonLinearLayer1)

        with tf.name_scope('outputLayer'):
            outputs = fully_connected_layer(hidden_layer, hiddenDim, inputOutputDim, nonlinearity = nonLinearLayer2)

        with tf.name_scope('error'):
            error = errorFunc(outputs, targets)

        with tf.name_scope('train_auto_encoder'):
            train_step = tf.train.AdamOptimizer().minimize(error)

        init = tf.global_variables_initializer()
    
    
    def trainAutoencoder(train_error, sess):
        for step, (input_batch, target_batch) in enumerate(dataProvider):
            _, batch_error = sess.run(
                [train_step, error], 
                feed_dict={inputs: input_batch, targets: target_batch})

            train_error += batch_error

        return train_error


    weights, biases = (None, None)

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init)

        totalRuntime = 0.

        prev_errors = np.zeros(3) #after three equalities break the while loop below

        train_error = -1

        #for e in range(numEpochs):
        e = 0
        while int(train_error * 1000) != int(np.mean(prev_errors)):
            train_error = 0.

            train_error, runTime = getRunTime(lambda : trainAutoencoder(train_error, sess))

            train_error /= dataProvider.num_batches

            totalRuntime += runTime

            if printing:
                print 'End epoch %02d (%.3f secs): err(train)=%.3f' % (e+1, runTime, train_error)

            prev_errors[0] = prev_errors[1]
            prev_errors[1] = prev_errors[2]
            #prev_errors[2] = np.round(train_error, 3)
            prev_errors[2] = int(train_error * 1000)

            e += 1
		
        weights, biases = [v.eval() for v in tf.trainable_variables() if "fullyConnected" in v.name]
        assert weights.shape[1] == len(biases)
    
    return graph, weights, biases, totalRuntime
