from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

import cifar10_utils
import cifar10_siamese_utils
from convnet import ConvNet
from siamese import Siamese
import datetime
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
X_TEST_BATCH_SIZE = 100;

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
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

FLAGS = None

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
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    save_path = FLAGS.checkpoint_dir + "/model.chpt"

    with tf.name_scope("Model"):
        convnet = ConvNet()

        x = tf.placeholder("float", (None, 32, 32, 3))
        y = tf.placeholder("float", (None, 10))
        avr_acc = tf.placeholder("float", (1))
        avr_loss = tf.placeholder("float", (1))

        predictions = convnet.inference(x)
        loss        = convnet.loss(predictions, y)
        accuracy    = convnet.accuracy(predictions, y)
        optimize    = tf.train.AdamOptimizer(FLAGS.learning_rate)
        minimize    = optimize.minimize(loss)
        merged      = tf.merge_all_summaries()
        log_test    = convnet._log_acc_loss(avr_acc, avr_loss)
        saver       = tf.train.Saver()


        timestamp = "/" + datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        with tf.Session() as sess:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + timestamp + '/train', sess.graph)
            test_writer  = tf.train.SummaryWriter(FLAGS.log_dir + timestamp + '/test')

            if tf.gfile.Exists(save_path):
                saver.restore(sess, save_path)
            else:

                sess.run(tf.initialize_all_variables())

                for batch_n in range(FLAGS.max_steps):
                    x_batch, y_batch = cifar10.train.next_batch(FLAGS.batch_size)

                    # Training
                    if batch_n % FLAGS.print_freq == 0 or batch_n == FLAGS.max_steps-1:
                        _, train_loss, train_acc, summary_train = sess.run([minimize, loss, accuracy, merged], {x:x_batch, y:y_batch})
                        train_writer.add_summary(summary_train, batch_n)
                        print("Iteration:", batch_n)
                        print("Train loss: ", train_loss )
                        print("Train acc: ", train_acc )
                    else:
                        _ = sess.run([minimize], {x:x_batch, y:y_batch})

                    # Testing
                    if batch_n % FLAGS.eval_freq == 0 or batch_n == FLAGS.max_steps-1:
                        x_test, y_test = cifar10.test.images, cifar10.test.labels

                        summary_test, test_loss, test_accuracy = sess.run([merged, loss, accuracy], {x:x_test, y:y_test})

                        test_writer.add_summary(summary_test, batch_n)
                        print("Test loss:    ", test_loss)
                        print("Test accuracy:", test_accuracy)
                        print("-------------------------")

                    # Checkpoints
                    if batch_n % FLAGS.checkpoint_freq == 0 or batch_n == FLAGS.max_steps-1:
                        #saver.save(sess, save_path)
                        print("SAVED MODEL")

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
    validation set using the data sampling function you implemented for siamese
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
    print("SIAMESE")
    cifar10 = cifar10_siamese_utils.get_cifar10('cifar10/cifar-10-batches-py')
    save_path = FLAGS.checkpoint_dir + "/siamese_model.chpt"

    with tf.name_scope("Model"):
        siamese = Siamese()
        margin = 1

        x_1 = tf.placeholder("float", (None, 32, 32, 3), name="x1")
        x_2 = tf.placeholder("float", (None, 32, 32, 3), name="x2")
        y = tf.placeholder("float", (None), name="y")

        predictions1 = siamese.inference(x_1)
        predictions2 = siamese.inference(x_2, True)
        loss        = siamese.loss(predictions1, predictions2, y, margin)
        #optimize    = tf.train.AdamOptimizer(FLAGS.learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.1*FLAGS.learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        minimize = optimizer.apply_gradients(capped_gvs)

        #minimize    = optimize.minimize(loss)
        merged      = tf.merge_all_summaries()
        saver       = tf.train.Saver()

        timestamp = "/" + datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        with tf.Session() as sess:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + timestamp + '/siamese_train', sess.graph)
            test_writer  = tf.train.SummaryWriter(FLAGS.log_dir + timestamp + '/siamese_test')

            if tf.gfile.Exists(save_path):
                saver.restore(sess, save_path)
            else:

                sess.run(tf.initialize_all_variables())

                for batch_n in range(FLAGS.max_steps):
                    print("iteration:", batch_n)
                    x1, x2, labels = cifar10.train.next_batch(FLAGS.batch_size)

                    # Training
                    if batch_n % FLAGS.print_freq == 0 or batch_n == FLAGS.max_steps-1:
                        _, train_loss, summary_train = sess.run([minimize, loss, merged], {x_1:x1, x_2:x2, y:labels})
                        train_writer.add_summary(summary_train, batch_n)
                        print("Iteration:", batch_n)
                        print("Train loss: ", train_loss )
                    else:
                        _ = sess.run(minimize, {x_1:x1, x_2:x2, y:labels})

                    '''
                    # Testing
                    if batch_n % FLAGS.eval_freq == 0 or batch_n == FLAGS.max_steps-1:
                        x_test, y_test = cifar10.test.images, cifar10.test.labels

                        summary_test, test_loss = sess.run([merged, loss], {x:x_test, y:y_test})

                        test_writer.add_summary(summary_test, batch_n)
                        print("Test loss:    ", test_loss)
                        print("-------------------------")
                    '''

                    # Checkpoints
                    if batch_n % FLAGS.checkpoint_freq == 0 or batch_n == FLAGS.max_steps-1:
                        saver.save(sess, save_path+str(batch_n))
                        print("SAVED MODEL")




    #test_batches = cifar10.create_dataset(?)
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
    print("FEATURE EXTRACTION")
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    save_path   = FLAGS.checkpoint_dir + "/model.chpt"

    if not tf.gfile.Exists(save_path):
        raise ValueError("you need to train the model first!")

    with tf.name_scope("Model"):
        convnet = ConvNet()

        x = tf.placeholder("float", (None, 32, 32, 3))
        predictions = convnet.inference(x)
        fc1         = tf.get_default_graph().get_tensor_by_name("Model/ConvNet/fc1/out:0")
        fc2         = tf.get_default_graph().get_tensor_by_name("Model/ConvNet/fc2/out:0")
        flatten     = tf.get_default_graph().get_tensor_by_name("Model/ConvNet/flatten/out:0")

        with tf.Session() as sess:
            saver   = tf.train.Saver()
            saver.restore(sess, save_path)

            x_test, y_test = cifar10.test.images[:5000], cifar10.test.labels[:5000]

            output_layers = [flatten, fc1, fc2]
            layers = sess.run(output_layers, {x:x_test})
            layer_names = ["flatten", "fc1", "fc2"]


            # TSNE
            plt.figure(figsize=(30,10))
            labels = np.argmax(y_test,1)
            color_map = cm.jet(np.linspace(0, 1, 10))

            colors = color_map[labels]
            for i,layer in enumerate(layers):
                plt.subplot(1,3,i+1)
                tsne = TSNE(learning_rate=1000).fit_transform(layer)
                plt.scatter(tsne[:,0], tsne[:,1], c=colors, s =50)
                plt.title(layer_names[i])
            plt.savefig("TSNE.png")

            #One VS Rest
            x_batch, y_batch = cifar10.train.next_batch(FLAGS.batch_size)
            for i, output_layer in enumerate(output_layers):
                classifier = OneVsRestClassifier(LinearSVC(random_state=0))
                features_test = sess.run(output_layer, {x:x_test})
                features_train = sess.run(output_layer, {x:x_batch})
                classifier = classifier.fit(features_train, y_batch)

                print(layer_names[i], "accuracy:", classifier.score(features_test, y_test))


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

    if FLAGS.is_train == "True":
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
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
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
    parser.add_argument('--is_train', type = str, default = "True",
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
