from __future__ import absolute_import
from __future__ import division

import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np

from model import Model

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    
    m_plus = 0.9
    m_minus = 0.1
    lambda_caps_loss = 0.5
    init_sigma_caps_W = 0.01
    caps2_num_caps = 10
    caps2_num_dims = 16
    epsilon=1e-7
    caps1_num_maps = 32
    caps1_num_caps = caps1_num_maps * 6 * 6  # 1152 primary capsules
    caps1_num_dims = 8
    num_routings = 3
    
    alpha_decoder_loss = 0.0005
    
    num_hidden1 = 512
    num_hidden2 = 1024
    x_flat_size = 28 * 28
    
        
    conv1_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu,
    }

    conv2_params = {
        "filters": caps1_num_maps * caps1_num_dims, # 256 convolutional filters
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
    }

    hidden_size = 200
    batch_size = 50
    num_epochs = 5 #10
    lr = 0.001
    """
    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/CapsNet/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"
    """

class CapsuleNetworksModel(Model):
    """
    Implements a Capsule network on Mnist dataset

    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, 28, 28, 1), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None,), type tf.int32
        mask_with_lables: A boolean placeholder to use either predictions or labels on image reconstruction

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_with_labels
        """

        
        self.input_placeholder = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="input_placeholder_X")
        self.mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
        self.labels_placeholder = tf.placeholder(tf.int32, shape = [None,], name = "labels_placeholder") 


    def create_feed_dict(self, inputs_batch, labels_batch, mask_with_labels = False):
        """Creates the feed_dict for the model.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            mask_with_label: a boolean value.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.labels_placeholder] = labels_batch
        feed_dict[self.mask_with_labels] = mask_with_labels #

        return feed_dict


    def add_prediction_op(self):
        """
        build the computational graph
        
        Add these variables to self as the instance variables
            self.correct
            self.y_predictions
            self.accuracy
        Returns:
            pred: tf.Tensor of shape (batch_size, caps2_num_caps, caps2_num_dim) (batch_size, 10, 16)
        """

        #X = self.input_placeholder
        with tf.name_scope("add_prediction_op"): 
            conv1 = tf.layers.conv2d(self.input_placeholder, name="conv1", **self.config.conv1_params)
            conv2 = tf.layers.conv2d(conv1, name="conv2", **self.config.conv2_params)

            #shape [batchsize, 1152, 8]
            caps1_raw = tf.reshape(conv2, [-1, self.config.caps1_num_caps, self.config.caps1_num_dims],
                           name="caps1_raw")
            caps1_output = squash(caps1_raw, name="caps1_output")

            self.caps2_output = capsule_routing(caps1_output, 1, 
                                           [self.config.caps1_num_caps, self.config.caps1_num_dims], 
                                           [self.config.caps2_num_caps, self.config.caps2_num_dims], 
                                           self.config.num_routings, self.config.init_sigma_caps_W)



            pred = self.caps2_output
            
            self.y_predictions = self.get_predictions(self.caps2_output)
            
            self.correct = tf.equal(self.labels_placeholder, self.y_predictions, name="correct")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name="accuracy")
                
            
            caps2_output_masked = self.mask_output(self.caps2_output)

            self.decoder_output = self.decoder(caps2_output_masked)
            

        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        
        Args:
            pred: A tensor of shape (batch_size, caps2_num_caps, caps2_num_dim) containing the capsule output.
        Returns:
            loss: A 0-d tensor (scalar)
        """

        with tf.name_scope("add_loss_op"):         

            
            margin_loss = self.loss_capsule(self.caps2_output)
            decoder_loss = self.loss_reconstruction(self.decoder_output)

            loss = tf.add(margin_loss, self.config.alpha_decoder_loss * decoder_loss, name="loss")
                                   
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        with tf.name_scope("train_op"): 
            train_op = tf.train.AdamOptimizer(learning_rate = self.config.lr).minimize(loss)

        return train_op

    def predict_on_batch(self, sess, inputs_batch):
        """
        Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (batch_size, 28, 28, 1)
        Returns:
            caps_output: np.ndarray of shape (batch_size, caps2_num_caps, caps2_num_dim)
            decoder_output: np.ndarray of shape (batch_size, 28*28*1)
            predictions: np.ndarray of shape (batch_size, )
        """
        with tf.name_scope("predict_on_batch"): 
            feed = self.create_feed_dict(inputs_batch, np.zeros([inputs_batch.shape[0],], dtype = np.int64), False)
            caps_output, decoder_output, predictions = sess.run([self.pred, self.decoder_output, self.y_predictions], feed_dict=feed)
            
        return caps_output, decoder_output, predictions
    
    def predict_on_tweaked_batch(self, sess, tweaked_vectors, tweaked_labels):
        """
        n_steps = 11

        tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
        tweaked_vectors_reshaped = tweaked_vectors.reshape(
            [-1, 1, caps2_n_caps, caps2_n_dims, 1])
        
        tweak_labels = np.tile(mnist.test.labels[:n_samples], caps2_n_dims * n_steps)
        """ 

        decoder_output_value = sess.run(
                self.decoder_output,
                feed_dict={self.caps2_output: tweaked_vectors,
                           self.mask_with_labels: True,
                           self.labels_placeholder: tweaked_labels})
        
        return decoder_output_value
    
    def validate_on_batch(self, sess, inputs_batch, labels_batch):
        """Test for the validate batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (batch_size, 28, 28, 1)
            labels_batch: np.ndarray of shape (batch_size, ) 
        Returns:
            caps_output: np.ndarray of shape (batch_size, caps2_num_caps, caps2_num_dim)
            decoder_output: np.ndarray of shape (batch_size, 28*28*1)
            predictions: np.ndarray of shape (batch_size, )
        """

        with tf.name_scope("validate_on_batch"): 
            feed = self.create_feed_dict(inputs_batch, labels_batch, False)
            predictions, p_correct, p_loss, p_accuracy = sess.run([self.y_predictions, self.correct, self.loss, self.accuracy], feed_dict=feed)
            
        return predictions, p_correct, p_loss, p_accuracy

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_with_labels):

        with tf.name_scope("train_on_batch"): 
            feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_with_labels = mask_with_labels)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
        
    def get_predictions(self, caps_output):
        with tf.name_scope("get_predictions"): 
                y_proba = safe_norm(caps_output, axis= -1, name = "y_proba")
                y_pred = tf.argmax(y_proba, axis=1, name="y_proba", output_type=tf.int32)
        return y_pred
        
    def mask_output(self, caps_output):
        """
        Mask output
        """
        with tf.name_scope("mask_output"): 
            y = self.labels_placeholder
            y_pred = self.y_predictions

            reconstruction_targets = tf.cond(self.mask_with_labels, # condition
                                             lambda: y,        # if True
                                             lambda: y_pred,   # if False
                                             name="reconstruction_targets")

            reconstruction_mask = tf.one_hot(reconstruction_targets,
                                             depth=self.config.caps2_num_caps,
                                             name="reconstruction_mask")

            reconstruction_mask_reshaped = tf.reshape(
                reconstruction_mask, [-1, self.config.caps2_num_caps, 1],
                name="reconstruction_mask_reshaped")

            caps_output_masked = tf.multiply(
                caps_output, reconstruction_mask_reshaped,
                name="caps_output_masked")

        return caps_output_masked
    

    
    def decoder(self, caps_output_masked):
        """
        add the decoder to the computational graph
        """

        with tf.name_scope("decoder"):            
            decoder_input = tf.reshape(caps_output_masked,
                               [-1, self.config.caps2_num_caps * self.config.caps2_num_dims],
                               name="decoder_input")
            hidden1 = tf.layers.dense(decoder_input, self.config.num_hidden1,
                                      activation=tf.nn.relu,
                                      name="hidden1")
            hidden2 = tf.layers.dense(hidden1, self.config.num_hidden2,
                                      activation=tf.nn.relu,
                                      name="hidden2")
            decoder_output = tf.layers.dense(hidden2, self.config.x_flat_size,
                                             activation=tf.nn.sigmoid,
                                             name="decoder_output")
        return decoder_output
    
    def loss_capsule(self, caps_output):
    
        """
        capsule loss
        """
        with tf.name_scope("loss_capsule"):         
            caps_output_norm = safe_norm(caps_output, axis=-1, keep_dims=False,
                                      name="caps_output_norm")


            T = tf.one_hot(self.labels_placeholder, depth = self.config.caps2_num_caps, name = "T")

            present_error = tf.square(tf.maximum(0., self.config.m_plus - caps_output_norm),
                                      name="present_error")

            absent_error = tf.square(tf.maximum(0., caps_output_norm - self.config.m_minus),
                                     name="absent_error")

            loss_raw = tf.add(T * present_error, self.config.lambda_caps_loss * (1.0 - T) * absent_error, name="loss_raw")

            loss_margin = tf.reduce_mean(tf.reduce_sum(loss_raw, axis=1), name="loss_margin")

        return loss_margin

    def loss_reconstruction(self, decoder_output):
        """
        reconstruction loss
        """
        with tf.name_scope("reconstruction_loss"): 
            X_flat = tf.reshape(self.input_placeholder, [-1, self.config.x_flat_size], name="X_flat")
            squared_difference = tf.square(X_flat - decoder_output,
                                           name="squared_difference")
            reconstruction_loss = tf.reduce_mean(squared_difference,
                                                name="reconstruction_loss")

        return reconstruction_loss
    
    
    def __init__(self, config):

        # Defining placeholders.
        self.config = config
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_with_labels = None
        self.caps2_output = None
        self.decoder_output = None
        self.y_predictions = None
        self.correct = None
        self.accuracy = None

        self.build()
    
def capsule_routing(caps1_output, lower_i, caps1_shape, caps2_shape, num_routings, init_sigma): 
    
    """
    #input 
    #caps1, 3-D array lower level capsule output, shape: [batch_size, caps1_num_caps, caps1_num_dim]
    #lower_i, scalar, input capsule level
    #caps1_shape: 1-D array [caps1_num_caps, caps1_num_dims]
    #cpas2_shape: 1-D array [caps2_num_caps, caps2_num_dims]
    #num_routings: integer number of routing loops
    #output: 3-D array [batch, caps2_num_caps, caps2_num_dim]
    """
    
    with tf.variable_scope("routing_" + str(lower_i) + "_"+str(lower_i + 1)):
        
        batch_size = tf.shape(caps1_output)[0]
        
        #print(batch_size)

        caps1_num_caps, caps1_num_dims = caps1_shape[0], caps1_shape[1]
        caps2_num_caps, caps2_num_dims = caps2_shape[0], caps2_shape[1]

        #matmul([16, 8] , [8, 1]) = [16, 1]
        W_init = tf.random_normal(
            shape=(1, caps1_num_caps, caps2_num_caps, caps2_num_dims, caps1_num_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init"+str(lower_i))
        W = tf.Variable(W_init, name="W"+str(lower_i))

        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled"+str(lower_i)) #for paralleling purpose 
        #W_tiled shape: [batch_size, caps1_num_caps, caps2_num_caps, caps2_num_dims, caps1_num_dims]
        #[batch_size, 1152, 10, 16, 8]
        
        #so the input of caps1 to matmul (caps1_output_tiled) with a shape: 
        #[batch_size, caps1_num_caps, caps2_num_caps, caps1_num_dims, 1]
        #[batch_size, 1152, 10, 8, 1]
        
        # the output(caps2) of the matmul with a shape:
        #[batch_size, caps1_num_caps, caps2_num_caps, caps2_num_dims, 1]
        #[batch_size, 1152, 10, 16, 1]
        
        #current caps1_output shape:[batch_size, caps1_num_caps, caps1_num_dims]
        #[batchsize, 1152, 8] to transfer to caps1_output_tiled
        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps" + str(lower_i) + "_output_expanded")
        #caps1_output_expanded [batch_size, 1152, 8, 1]
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                           name="caps" + str(lower_i) + "_output_tile")
        #caps1_output_tile [batch_size, 1152,1, 8, 1]
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_num_caps, 1, 1],
                                     name="caps" + str(lower_i) + "_output_tiled")
        #caps1_output_tiled [batch_size, 1152,10, 8, 1]
        
        #caps2_output_tiled [batch_size, 1152, 10, 16, 1]
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name = "Caps" + str(lower_i + 1) + "_predicted")

        #Routing
        
        #b_i_j [batch_size, 1152, 10, 1, 1] [batch_size, caps1_num_caps, caps2_num_caps, 1,1]
        b_i_j = tf.zeros([batch_size, caps1_num_caps, caps2_num_caps, 1, 1],
                               dtype=np.float32, name="b_" + str(lower_i) + "_" + str(lower_i + 1))
        
        for i in range(num_routings):
            #c_i_j [batch_size, 1152, 10, 1, 1] [batch_size, caps1_num_caps, caps2_num_caps, 1,1]
            c_i_j = tf.nn.softmax(b_i_j, dim=2, name="c_" + str(lower_i) + "_" + str(lower_i + 1) + "round_" + str(i))

            #weighted_predictions: [batch_size, 1152, 10, 16, 1]
            weighted_predictions = tf.multiply(c_i_j, caps2_predicted,
                                       name="weighted_predictions_round_" + str(i))
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                         name="weighted_sum_round_" + str(i))

            #weighted_sum [batch_size, 1, 10, 16, 1]

            caps2_output = squash(weighted_sum, axis=-2,
                                  name="caps2_output_round_" + str(i))
            #caps2_out [batch_size, 1, 10, 16, 1]
            
            if(i == num_routings - 1):
                caps2_output = tf.squeeze(caps2_output, [1, 4])
                break

            caps2_output_tiled = tf.tile(caps2_output, [1, caps1_num_caps, 1, 1, 1],
                name="caps2_output_tiled_round_" + str(i))
            #caps2_output_round_1_tiled [1152, 10, 16, 1]

            agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a = True, name="agreement_round_" +str(i))
            #agreement [batchsize, 1152, 10, 1, 1]

            b_i_j = tf.add(b_i_j, agreement, name="b_" + str(lower_i) + "_" + str(lower_i+1) + "_round_" + str(i+1))
            
        return caps2_output

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    dims = output_vectors.shape[2]
    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(dims) # 0, 1, ..., 15
    tweaks = np.zeros([dims, n_steps, 1, 1, dims])
    tweaks[pose_parameters, :, 0, 0, pose_parameters] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded