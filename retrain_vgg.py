from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import vgg
import cifar10_utils
import sys

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################

    optimizer = tf.train.AdamOptimizer()
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="classifier")
    train_op = optimizer.minimize(loss, var_list=weights)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    
    
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    weight_init_scale = 0.001
    cifar10 = cifar10_utils.get_cifar10(validation_size=500)


    
    x_in = tf.placeholder(tf.float32, [None,32,32,3])
    y_true = tf.placeholder(tf.float32, [None,10])
    
    with tf.variable_scope("classifier",reuse=None):

                        
        W1=tf.get_variable("W1",initializer=tf.random_normal([2048,384], stddev=weight_init_scale, dtype=tf.float32))
        W2=tf.get_variable("W2", initializer= tf.random_normal([384, 192], stddev=weight_init_scale, dtype=tf.float32))
        W3=tf.get_variable("W3", initializer = tf.random_normal([192,10], stddev=weight_init_scale, dtype=tf.float32))
    
    
    sess = tf.Session()
    saver = tf.train.Saver()
    #define things

 
    
    
    p5, assign_ops = vgg.load_pretrained_VGG16_pool5(x_in)

    pool5 = tf.stop_gradient(p5)
    
    flatten = tf.contrib.layers.flatten(pool5)

    fc1 = tf.nn.relu(tf.matmul(flatten, W1)) 
    fc2 = tf.nn.relu(tf.matmul(fc1, W2))
    logits = tf.matmul(fc2, W3)    
 
        
    
    celoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_true))
    #weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="classifier")
    #reg_loss = tf.reduce_mean(tf.contrib.layers.apply_regularization(
    #    tf.contrib.layers.l2_regularizer(1e-7), weights_list=weights))
            
    loss = celoss 
    
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(logits, 1), 
                tf.argmax(y_true, 1)),
            dtype = tf.float32))
    
    opt_iter = train_step(loss)
 


    

    xbat, ybat = cifar10.train.next_batch(100)
    
    #begin the training
    sess.run(tf.initialize_all_variables())
    sess.run(assign_ops)
    with sess:
        swriter = tf.train.SummaryWriter(FLAGS.log_dir + "\vgg");
        # loop
        for i in range(FLAGS.max_steps+1):
            xbat, ybat = cifar10.train.next_batch(FLAGS.batch_size)
            sess.run(opt_iter, feed_dict={x_in:xbat, y_true:ybat})
            if i % FLAGS.print_freq == 0:
                xbat, ybat = cifar10.validation.next_batch(100)
                val_acc, val_loss = sess.run([acc,loss], feed_dict={x_in:xbat, y_true:ybat})
                
                sys.stderr.write("iteration : " + str(i)
                      + ", validation loss : " 
                      + str(val_loss)
                      + ", validation_accuracy"
                      + str(val_acc) 
                                 + "\n")
                swriter.add_summary(sess.run(tf.scalar_summary("accuracy", val_acc)), i)

                
            if i% FLAGS.checkpoint_freq == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 
                                                  "iteration" + str(i) + ".ckpt"))
            if i%FLAGS.eval_freq ==0:
                xbat, ybat = cifar10.test.next_batch(100)
        
                sys.stderr.write("test accuracy:" + str(sess.run(acc, feed_dict={x_in:xbat, y_true:ybat})) + "\n")
    
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
