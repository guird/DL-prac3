from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import ConvNet
import sklearn

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

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
    opt_step = optimizer.minimize(loss)    
    return opt_step
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    #first lets test that out model works:
    
    #initialize:
    
    weight_init_scale = 0.001
    cifar10 = cifar10_utils.get_cifar10(validation_size=100)

    cnet = ConvNet(10)
    
    x_in = tf.placeholder(tf.float32, [None,32,32,3])
    y_true = tf.placeholder(tf.float32, [None,10])
    
    with tf.variable_scope("ConvNet",reuse=None):
        filter1=tf.get_variable("filter1",initializer=tf.random_normal([5,5,3,64], stddev=weight_init_scale, dtype=tf.float32))
        filter2=tf.get_variable("filter2",initializer=tf.random_normal([5,5,64,64], stddev=weight_init_scale, dtype=tf.float32))

                        
        W1=tf.get_variable("W1",initializer=tf.random_normal([4096,384], stddev=weight_init_scale, dtype=tf.float32))
        W2=tf.get_variable("W2", initializer= tf.random_normal([384, 192], stddev=weight_init_scale, dtype=tf.float32))
        W3=tf.get_variable("W3", initializer = tf.random_normal([192,10], stddev=weight_init_scale, dtype=tf.float32))
    
    
    sess = tf.Session()
    saver = tf.train.Saver()
    #define things
    logits = cnet.inference(x_in)
    loss= cnet.loss(logits,y_true)
    acc = cnet.accuracy(logits, y_true)
    opt_iter = train_step(loss)
    sess.run(tf.initialize_all_variables())
    

    

    #xbat, ybat = cifar10.train.next_batch(100)
    
    #begin the training
    with sess:
    
        # loop
        for i in range(FLAGS.max_steps):
            xbat, ybat = cifar10.train.next_batch(FLAGS.batch_size)
            sess.run(opt_iter, feed_dict={x_in:xbat, y_true:ybat})
            if i % FLAGS.print_freq == 0:
                xbat, ybat = cifar10.validation.next_batch(100)
                val_acc, val_loss = sess.run([acc,loss], feed_dict={x_in:xbat, y_true:ybat})
                
                sys.stderr.write("iteration : " + str(i)
                      + ", validation loss : " 
                      + str(val_loss)
                      + ", validation_accuracy"
                      + str(val_acc))
        
            if i% FLAGS.checkpoint_freq == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 
                                              "iteration" + str(i) + ".ckpt"))
        
        xbat, ybat = cifar10.test.next_batch(100)
        
        sys.stderr.write("test accuracy:" + str(sess.run(acc, feed_dict={x_in:xbat, y_true:ybat})))
    
    
    
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    sess = tf.Session()
    loader=tf.train.import_meta_graph(os.path.join(FLAGS.checkpoint_dir,'iteration' + str(i) + ".ckpt"))
    
    loader.restore(sess, tf.train.latest_checkpoint('./'))
    
    

    feats = sklearn.manifold.TSNE(n_components=10, random_state=0)
                                                   
    #######################
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

    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
