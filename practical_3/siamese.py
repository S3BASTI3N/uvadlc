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

    def __init__(self):
        self.weight_regularizer = tf.contrib.layers.regularizers.l2_regularizer(0.0001)
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.001 )

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
        with tf.variable_scope('ConvNet') as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            if reuse:
                conv_scope.reuse_variables()

            conv1 = self._conv_layer(x, 1, 3, 64, reuse=reuse )
            conv2 = self._conv_layer(conv1, 2, 64, 64, reuse=reuse )

            flatten = self._flatten_layer(conv2)

            fc1 = self._fully_connected_layer(flatten, 1, 4096, 384, tf.nn.relu, reuse=reuse)
            l2_out = fc2 = self._fully_connected_layer(fc1, 2, 384, 192, tf.nn.l2_normalize, dim=1, reuse=reuse)
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def _conv_layer(self, x, i, filter_depth, output_depth, filter_size=[5, 5], filter_stride=[1, 1],
                                                            pool_size=[3, 3], pool_stride=[2, 2], reuse=False):
        with tf.variable_scope("conv"+str(i)):
            kernel = tf.get_variable("kernel", (filter_size[0], filter_size[1], filter_depth, output_depth),
                                            initializer=self.initializer,
                                            regularizer=self.weight_regularizer)

            if i == 1 and not reuse:
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

    def _flatten_layer(self, x, reuse=False):
        with tf.variable_scope('flatten'):
            return tf.identity(tf.contrib.layers.flatten(x, None), name="out")


    def _fully_connected_layer(self, x, i, input_size, output_size, activation=None, dim=-1, reuse=False):
        with tf.variable_scope('fc'+str(i)):
            w = tf.get_variable("weight", (input_size, output_size),
                                            initializer=self.initializer,
                                            regularizer=self.weight_regularizer)
            b = tf.get_variable("bias", [1,output_size], initializer=tf.constant_initializer(0.0),
                                                            regularizer=self.weight_regularizer)

            if not reuse:
                tf.histogram_summary(w.name, w)
                tf.histogram_summary(b.name, b)
            mat = tf.add(tf.matmul(x, w),b)

            if dim >= 0:
                 return tf.identity(activation(mat, dim), name='out')
            if activation:
                return tf.identity(activation(mat), name='out')
            else:
                return tf.identity(mat,name='out')

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
        label = tf.reshape(label, [128,1])
        
        d  = tf.pow(tf.sub(channel_1, channel_2), 2)
        t1 = tf.matmul(label, d, transpose_a=True)
        t2 = tf.matmul(tf.sub(1.0, label), tf.maximum(tf.sub(margin, d), 0), transpose_a=True)
        loss = tf.reduce_mean(tf.add(t1, t2))
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
