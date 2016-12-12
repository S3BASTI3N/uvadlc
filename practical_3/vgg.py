import tensorflow as tf
import numpy as np

VGG_FILE = './pretrained_params/vgg16_weights.npz'

def load_pretrained_VGG16_pool5(input, scope_name='vgg'):
    """
    Load an existing pretrained VGG-16 model.
    See https://www.cs.toronto.edu/~frossard/post/vgg16/

    Args:
        input:         4D Tensor, Input data
        scope_name:    Variable scope name

    Returns:
        pool5: 4D Tensor, last pooling layer
        assign_ops: List of TF operations, these operations assign pre-trained values
                    to all parameters.
    """

    with tf.variable_scope(scope_name):

        vgg_weights, vgg_keys = load_weights(VGG_FILE)

        assign_ops = []
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            vgg_W = vgg_weights['conv1_1_W']
            vgg_B = vgg_weights['conv1_1_b']
            kernel = tf.get_variable('conv1_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_1/' + "biases", vgg_B.shape,
                initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)


        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            vgg_W = vgg_weights['conv1_2_W']
            vgg_B = vgg_weights['conv1_2_b']
            kernel = tf.get_variable('conv1_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))

            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv1_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            vgg_W = vgg_weights['conv2_1_W']
            vgg_B = vgg_weights['conv2_1_b']
            kernel = tf.get_variable('conv2_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            vgg_W = vgg_weights['conv2_2_W']
            vgg_B = vgg_weights['conv2_2_b']
            kernel = tf.get_variable('conv2_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv2_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            vgg_W = vgg_weights['conv3_1_W']
            vgg_B = vgg_weights['conv3_1_b']
            kernel = tf.get_variable('conv3_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            vgg_W = vgg_weights['conv3_2_W']
            vgg_B = vgg_weights['conv3_2_b']
            kernel = tf.get_variable('conv3_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer()
                                     )

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            vgg_W = vgg_weights['conv3_3_W']
            vgg_B = vgg_weights['conv3_3_b']
            kernel = tf.get_variable('conv3_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv3_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            vgg_W = vgg_weights['conv4_1_W']
            vgg_B = vgg_weights['conv4_1_b']
            kernel = tf.get_variable('conv4_1/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_1/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            vgg_W = vgg_weights['conv4_2_W']
            vgg_B = vgg_weights['conv4_2_b']
            kernel = tf.get_variable('conv4_2/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_2/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            vgg_W = vgg_weights['conv4_3_W']
            vgg_B = vgg_weights['conv4_3_b']
            kernel = tf.get_variable('conv4_3/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv4_3/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            vgg_W = vgg_weights['conv5_1_W']
            vgg_B = vgg_weights['conv5_1_b']
            kernel = tf.get_variable('conv5_1/'  + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_1/'  + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)


        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            vgg_W = vgg_weights['conv5_2_W']
            vgg_B = vgg_weights['conv5_2_b']
            kernel = tf.get_variable('conv5_2/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_2/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)


        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            vgg_W = vgg_weights['conv5_3_W']
            vgg_B = vgg_weights['conv5_3_b']
            kernel = tf.get_variable('conv5_3/' + 'weights', vgg_W.shape,
                                     initializer=tf.random_normal_initializer())

            assign_ops.append(kernel.assign(vgg_W))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('conv5_3/' + "biases", vgg_B.shape,
                                    initializer=tf.constant_initializer(0.0))

            assign_ops.append(biases.assign(vgg_B))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)


        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')
        print("pool5.shape: %s" % pool5.get_shape())

    return pool5, assign_ops

def load_weights(weight_file):
  weights = np.load(weight_file)
  keys = sorted(weights.keys())
  return weights, keys


class VGG(object):
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

    def _fully_connected(self, name, x, kernel_shape, bias_shape):
        with tf.variable_scope(name):
            W = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(stddev=0.001))
            b = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
            return tf.matmul(x, W) + b

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
        with tf.variable_scope('VGG'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            pool5, assign_ops = load_pretrained_VGG16_pool5(x, scope_name='VGG')
            flatten = tf.reshape(pool5, shape=[-1,512])
            fc1 = tf.nn.relu(self._fully_connected('fc1', flatten, [flatten.get_shape()[1], 384], [384]))
            fc2 = tf.nn.relu(self._fully_connected('fc2', fc1, [384, 192], [192]))
            logits = self._fully_connected('fc3', fc2, [192, 10], [10])
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits, assign_ops

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
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('Accuracy', accuracy)
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
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        #reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #loss = cross_entropy_loss + reg_loss
        tf.scalar_summary('Loss', loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
