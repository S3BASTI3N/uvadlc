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
        self.weight_regularizer = tf.contrib.layers.regularizers.l2_regularizer(0.0001)
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001 )

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
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            conv1 = self._conv_layer(x, 1, 3, 64 )
            conv2 = self._conv_layer(conv1, 2, 64, 64 )

            flatten = self._flatten_layer(conv2)

            fc1 = self._fully_connected_layer(flatten, 1, 4096, 384, tf.nn.relu)
            fc2 = self._fully_connected_layer(fc1, 2, 384, 192, tf.nn.relu)
            logits = self._fully_connected_layer(fc2, 3, 192, self.n_classes)

            ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def _log_acc_loss(self, acc, loss):
        tf.scalar_summary("test acc", acc)
        tf.scalar_summary("test loss", loss)

    def _conv_layer(self, x, i, filter_depth, output_depth, filter_size=[5, 5], filter_stride=[1, 1],
                                                            pool_size=[3, 3], pool_stride=[2, 2]):
        with tf.variable_scope("conv"+str(i)):
            kernel = tf.get_variable("kernel", (filter_size[0], filter_size[1], filter_depth, output_depth),
                                            initializer=self.initializer,
                                            regularizer=self.weight_regularizer)

            if i == 1:
                grid = self._put_kernels_on_grid(kernel, 8, 8)
                tf.image_summary("filter"+str(i), grid)

            conv = tf.nn.conv2d(x, kernel, [1, filter_stride[0], filter_stride[1], 1], "SAME")
            relu = tf.nn.relu(conv)
            return tf.nn.max_pool(relu, [1, pool_size[0], pool_size[1], 1], [1, pool_stride[0], pool_stride[1], 1], "SAME" )

    def _put_kernels_on_grid(self, kernel, grid_Y, grid_X, pad=1):
        '''
        Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)

        Return:
          Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
        '''
        # pad X and Y
        x1 = tf.pad(kernel, [[pad,0],[pad,0],[0,0],[0,0]] )

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + pad
        X = kernel.get_shape()[1] + pad

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 1]
        x_min = tf.reduce_min(x7)
        x_max = tf.reduce_max(x7)
        x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def _flatten_layer(self, x):
        with tf.variable_scope('flatten'):
            return tf.identity(tf.contrib.layers.flatten(x, None), name="out")


    def _fully_connected_layer(self, x, i, input_size, output_size, activation=None):
        with tf.variable_scope('fc'+str(i)):
            w = tf.get_variable("weight", (input_size, output_size),
                                            initializer=self.initializer,
                                            regularizer=self.weight_regularizer)
            b = tf.get_variable("bias", [1,output_size], initializer=tf.constant_initializer(0.0),
                                                            regularizer=self.weight_regularizer)

            tf.histogram_summary(w.name, w)
            tf.histogram_summary(b.name, b)
            mat = tf.add(tf.matmul(x, w),b)

            if activation:
                return tf.identity(activation(mat), name='out')
            else:
                return tf.identity(mat,name='out')

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
        batch_size = logits.get_shape()[0]

        correct = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        tf.scalar_summary("Accuracy", accuracy)
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
        weight_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        if weight_loss == None:
            weight_loss = 0
        cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        loss = weight_loss + cross_loss
        tf.scalar_summary("loss", loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
