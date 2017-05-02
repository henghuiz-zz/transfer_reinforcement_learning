import tensorflow as tf


class StudentLearner(object):
    def __init__(self, network, optimizer):
        # Placeholders
        self.input_x = network.x
        with tf.variable_scope(network.scope):
            self.input_y = tf.placeholder(
                'float', [None, network.number_action], name='Input_Action')
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.input_y, logits=network.logits
        )
        self.train_step = optimizer.minimize(self.loss)