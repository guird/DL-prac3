from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet',reuse=True):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            filter1 = tf.get_variable("filter1",dtype=tf.float32)
            filter2 = tf.get_variable("filter2",dtype=tf.float32)
            W1 = tf.get_variable("W1",dtype=tf.float32)
            W2 = tf.get_variable("W2",dtype=tf.float32)
            W3 = tf.get_variable("W3",dtype=tf.float32)
            
            #start with conv layers
            conv1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(x, filter1,  [1,1,1,1], "SAME" )),[1,3,3,1], [1,2,2,1], "SAME")
            
            conv2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(conv1, filter2, [1,1,1,1], "SAME")), [1,3,3,1], [1,2,2,1],"SAME")

            flatten = tf.contrib.layers.flatten(conv2)

            fc1 = tf.nn.relu(tf.matmul(flatten, W1)) 
            fc2 = tf.nn.relu(tf.matmul(fc1, W2))
            fc3 = tf.matmul(fc2, W3)
            
            logits = fc3

            
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(logits, 1), 
                    tf.argmax(labels, 1)),
                dtype = tf.float32))
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################

        celoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ConvNet")
        reg_loss = tf.reduce_mean(tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(1e-7), weights_list=weights))
            
        loss = tf.add(celoss, reg_loss)
        
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
