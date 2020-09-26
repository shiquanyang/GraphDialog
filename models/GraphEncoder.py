import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.ops import embedding_ops
from utils.config import *
from models.Libraries.BidirectionalGraphEncoder import *


class GraphEncoder(tf.keras.Model):
    def __init__(self, input_size, hidden_size, dropout, lang, recurrent_size, n_layers=1):
        super(GraphEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.recurrent_size = recurrent_size
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.embedding = tf.keras.layers.Embedding(input_size, hidden_size, embeddings_initializer=tf.initializers.RandomNormal(0.0, 1.0))  # different: pad token embedding mask.
        self.bi_graph_gru = BidirectionalGraphEncoder(hidden_size, hidden_size, lang.n_types, recurrent_size)
        self.W = tf.keras.layers.Dense(hidden_size,
                                       use_bias=True,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size)))
                                       )
        self.W1 = tf.keras.layers.Dense(2*hidden_size,
                                       use_bias=True,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size)))
                                       )
        self.W2 = tf.keras.layers.Dense(4*hidden_size,
                                       use_bias=True,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size)))
                                       )
        self.W3 = tf.keras.layers.Dense(hidden_size,
                                       use_bias=True,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size)))
                                       )
        self.softmax = tf.keras.layers.Softmax(1)
        self.relu = tf.keras.layers.ReLU()

    def initialize_hidden_state(self, batch_size):
        forward_hidden = tf.zeros((self.recurrent_size, batch_size, self.hidden_size))
        backward_hidden = tf.zeros((self.recurrent_size, batch_size, self.hidden_size))
        return [forward_hidden, backward_hidden]

    def gen_input_mask(self, batch_size, max_len, lengths):
        input_mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for id, len in enumerate(lengths):
            input_mask[id, :lengths[id]] = np.ones([1, lengths[id]], dtype=np.float32)
        return tf.convert_to_tensor(input_mask)

    def gen_embedding_mask(self, input):
        raw_mask_array = [[1.0]] * PAD_token + [[0.0]] + [[1.0]] * (self.input_size - PAD_token - 1)
        mask = embedding_ops.embedding_lookup(raw_mask_array, tf.cast(input, dtype=tf.int32))
        ret_mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, self.hidden_size])
        return ret_mask

    def call(self, input_seqs, input_lengths, deps, edge_types, cell_mask, hidden=None, training=True):
        # batch_size = input_seqs.shape[0]
        # max_len = input_seqs.shape[1]
        mask = self.gen_input_mask(input_seqs.shape[0], input_seqs.shape[1], input_lengths)
        embedded = self.embedding(tf.reshape(input_seqs, [input_seqs.get_shape()[0], -1]))  # different: pad token embedding not masked. input_seqs: batch_size * input_length * MEM_TOKEN_SIZE.
        pad_mask = self.gen_embedding_mask(tf.reshape(input_seqs,[input_seqs.shape[0], -1]))
        embedded = tf.multiply(embedded, pad_mask)
        embedded = tf.reshape(
            embedded, [input_seqs.get_shape()[0], input_seqs.get_shape()[1], input_seqs.get_shape()[2], embedded.get_shape()[-1]])  # embedded: batch_size * input_length * MEM_TOKEN_SIZE * embedding_dim.
        embedded = tf.math.reduce_sum(embedded, 2)  # embedded: batch_size * input_length * embedding_dim.
        if training:
            embedded = self.dropout_layer(embedded, training=training)
        hidden = self.initialize_hidden_state(input_seqs.get_shape()[0])
        outputs, hidden_f, hidden_b = self.bi_graph_gru(embedded,
                                                        input_lengths,
                                                        tf.transpose(deps, [1, 0, 2, 3]),
                                                        tf.transpose(edge_types, [1, 0, 2, 3]),
                                                        mask,
                                                        tf.transpose(cell_mask, [1, 0, 2, 3]),
                                                        hidden,
                                                        training)  # outputs: batch_size*max_len*(2*embedding_dim)
        hidden_hat = tf.concat([hidden_f, hidden_b], 1)
        hidden = self.W(hidden_hat)
        outputs = self.W(outputs)
        return outputs, hidden