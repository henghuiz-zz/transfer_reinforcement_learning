import tensorflow as tf


def prelu(_x):
    """
    The Parametric Rectified Linear Unit
    :param _x: input tensor 
    :return: output of the RPeLu
    """
    alphas = tf.get_variable('alpha', [],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32, trainable=False)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - tf.abs(_x)) * 0.5

    return pos + neg


class CNNTypeOne:
    """
    Type one neural network used in this course project.
    """

    def __init__(self, game_name, number_action, trainable, prefix='teacher_'):
        self.game_name = game_name
        self.number_action = number_action
        self.scope = prefix+game_name

        with tf.variable_scope(self.scope):
            self.x = tf.placeholder(
                tf.float32, [None, 84, 84, 12], name='Input_Images')
            conv1 = tf.layers.conv2d(self.x, 32, 5, padding='same',
                                     name='conv0', trainable=trainable,
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='Max_Pool_1')
            conv2 = tf.layers.conv2d(pool1, 32, 5, padding='same',
                                     name='conv1', trainable=trainable,
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='Max_Pool_2')
            conv3 = tf.layers.conv2d(pool2, 64, 4, padding='same',
                                     name='conv2', trainable=trainable,
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='Max_Pool_3')
            conv4 = tf.layers.conv2d(pool3, 64, 3, padding='same',
                                     name='conv3', trainable=trainable,
                                     activation=tf.nn.relu)

            flatten = tf.contrib.layers.flatten(conv4)

            fc1 = tf.layers.dense(flatten, units=512, activation=tf.identity,
                                  name='fc0', trainable=trainable)
            with tf.variable_scope('prelu'):
                active_1 = prelu(fc1)

            self.logits = tf.layers.dense(
                inputs=active_1, units=self.number_action, trainable=trainable,
                activation=None, name='fc-pi')
            self.policy = tf.nn.softmax(self.logits)

            self.value = tf.layers.dense(
                inputs=active_1, units=1, trainable=trainable,
                activation=None, name='fc-v'
            )

        # Define saver for all the variable
        self.variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope
        )
        self.saver = tf.train.Saver(self.variables)

    def load(self, sess, path):
        self.saver.restore(sess, path)

    def save_variable(self, sess, path, step):
        self.saver.save(sess, path, global_step=step, write_meta_graph=False)


class CNNTypeTwo:
    """
    Type two neural network used in this course project.
    """

    def __init__(self, game_name, number_action, trainable, prefix='student_'):
        self.game_name = game_name
        self.number_action = number_action
        self.scope = prefix + game_name

        with tf.variable_scope(self.scope):
            self.x = tf.placeholder(
                tf.float32, [None, 84, 84, 12], name='Input_Images')
            conv1 = tf.layers.conv2d(self.x, 64, 5, padding='same',
                                     name='conv0', trainable=trainable,
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='Max_Pool_1')
            conv2 = tf.layers.conv2d(pool1, 64, 5, padding='same',
                                     name='conv1', trainable=trainable,
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='Max_Pool_2')
            conv3 = tf.layers.conv2d(pool2, 64, 4, padding='same',
                                     name='conv2', trainable=trainable,
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='Max_Pool_3')

            flatten = tf.contrib.layers.flatten(pool3)

            self.fc_1 = tf.layers.dense(
                flatten, units=256, activation=tf.nn.relu, name='fc0',
                trainable=trainable)
            self.fc_end = tf.layers.dense(
                self.fc_1, units=256, activation=tf.nn.relu, name='fc0',
                trainable=trainable)

            self.logits = tf.layers.dense(
                inputs=self.fc_end, units=self.number_action,
                trainable=trainable, activation=None, name='fc-pi')
            self.policy = tf.nn.softmax(self.logits)

            self.value = tf.layers.dense(
                inputs=self.fc_end, units=1, trainable=trainable,
                activation=None, name='fc-v'
            )

        # Define saver for all the variable
        self.variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope
        )
        self.saver = tf.train.Saver(self.variables)

    def load(self, sess, path):
        self.saver.save(sess, path)

    def save_variable(self, sess, path, step):
        self.saver.save(sess, path, global_step=step, write_meta_graph=False)