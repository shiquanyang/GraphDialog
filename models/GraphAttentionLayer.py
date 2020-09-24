import tensorflow as tf
import numpy as np
import pdb


class GraphAttentionLayer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.W = tf.keras.layers.Dense(
            output_dim,
            use_bias=True,
            kernel_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(2 * output_dim)), (1 / np.sqrt(2 * output_dim))),
            bias_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(2 * output_dim)), (1 / np.sqrt(2 * output_dim))))  # different: bias should be explicitly assigned.
        # self.a = self.add_weight(
        #     name='a',
        #     shape=(self.output_dim * 2, 1),
        #     initializer='glorot_uniform',
        #     regularizer=None,
        #     constraint=None
        # )  # self.a: (output_dim * 2) * 1
        self.a1 = self.add_weight(
            name='a1',
            shape=(self.output_dim, 1),
            initializer='glorot_uniform',
            # initializer=tf.initializers.RandomUniform(-(1/np.sqrt(self.output_dim)),(1/np.sqrt(self.output_dim))),
            # initializer='glorot_normal',
            regularizer=None,
            constraint=None
        )  # self.a1: (output_dim, 1)
        self.a2 = self.add_weight(
            name='a2',
            shape=(self.output_dim, 1),
            initializer='glorot_uniform',
            # initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(self.output_dim)), (1 / np.sqrt(self.output_dim))),
            # initializer='glorot_normal',
            regularizer=None,
            constraint=None
        )  # self.a2: (output_dim, 1)
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.elu = tf.keras.layers.ELU()
        self.softmax = tf.keras.layers.Softmax(2)

    def call(self, input, adj, training=True):  # input: batch_size * max_len * embedding_dim, adj: batch_size * max_len * max_len.
        # add for mimic memory
        # h = self.W(input)  # h: batch_size * max_len * output_dim.
        h = tf.identity(input)
        batch_size, N = h.shape[0], h.shape[1]  # batch_size: batch_size, N: number of nodes in graph.
        # f_1 = h @ self.a1
        # f_2 = h @ self.a2
        f_1 = tf.matmul(h, tf.tile(tf.expand_dims(self.a1, 0), [batch_size, 1, 1]))
        f_2 = tf.matmul(h, tf.tile(tf.expand_dims(self.a2, 0), [batch_size, 1, 1]))
        prob_logits = self.leakyrelu(tf.tile(f_1, [1, 1, N]) + tf.tile(tf.transpose(f_2, [0, 2, 1]), [1, N, 1]))
        prob_logits = tf.where(adj > 0, prob_logits, (-1 * np.ones_like(prob_logits) * np.inf))  # prob_logits: batch_size * max_len * max_len.
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * max_len * max_len.
        # comment for mimic memory
        # if training:
        #     prob_soft = self.dropout_layer(prob_soft, training=training)  # prob_soft: batch_size * max_len * max_len.
        h_prime = tf.matmul(prob_soft, h)  # h_prime: batch_size * max_len * output_dim.

        if self.concat:
            return self.elu(h_prime)
            # add for mimic memory
            # return h_prime
        else:
            return h_prime