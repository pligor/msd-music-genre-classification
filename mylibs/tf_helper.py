# -*- coding: utf-8 -*-
"""Tensorflow Helper"""

import tensorflow as tf
from py_helper import merge_dicts

def tfMSE(outputs, targets):
    #return tf.reduce_mean(tf.square(tf.sub(targets, outputs)))
    return tf.reduce_mean(tf.squared_difference(targets, outputs))

def tfRMSE(outputs, targets):
    return tf.sqrt(tfMSE(outputs, targets))


def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu, avoidDeadNeurons=0.,
                          w=None, b=None):
    weights = tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5) if w is None else w,
            'weights'
    )
    
    biases = tf.Variable(avoidDeadNeurons * tf.ones([output_dim]) if b is None else b, 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs


def trainEpoch(inputs, targets, sess, train_data, train_step, error, accuracy, extraFeedDict = {}):
    train_error = 0.
    train_accuracy = 0.
    
    num_batches = train_data.num_batches
    
    for step, (input_batch, target_batch) in enumerate(train_data):
        _, batch_error, batch_acc = sess.run(
            [train_step, error, accuracy], 
            feed_dict= merge_dicts({inputs: input_batch, targets: target_batch}, extraFeedDict)
        )
        
        train_error += batch_error
        train_accuracy += batch_acc
    
    train_error /= num_batches
    
    train_accuracy /= num_batches
    
    return train_error, train_accuracy


def validateEpoch(inputs, targets, sess, valid_data, error, accuracy, keep_prob_keys = [], extraFeedDict={}):
    valid_error = 0.
    valid_accuracy = 0.
    
    num_batches = valid_data.num_batches
    
    validationKeepProbability = 1. #it is always 100% for validation
    
    keep_prob_dict = dict()
    
    for keep_prob_key in keep_prob_keys:
        keep_prob_dict[keep_prob_key] = validationKeepProbability

    for step, (input_batch, target_batch) in enumerate(valid_data):
        batch_error, batch_acc = sess.run(
            [error, accuracy], 
            feed_dict= merge_dicts({inputs: input_batch, targets: target_batch}, keep_prob_dict, extraFeedDict)
        )
            
        valid_error += batch_error
        valid_accuracy += batch_acc

    valid_error /= num_batches
    valid_accuracy /= num_batches
    
    return valid_error, valid_accuracy
