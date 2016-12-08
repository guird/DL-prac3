from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('Siamese', reuse = True) as conv_scope:
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
            l2_out = tf.nn.l2_normalize(fc2, 1)
            



            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        Y = tf.cast(label, tf.float32)
        d2 = tf.reduce_sum(tf.square(tf.sub(channel_1, channel_2)),1)
        loss = tf.reduce_sum(tf.add(tf.mul(Y , d2), 
                      tf.mul(tf.sub(1.0, Y), tf.maximum
                             (margin,d2))))
                                                       

        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
