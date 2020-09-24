import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.ops import embedding_ops
from utils.config import *

class ContextRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.embedding = tf.keras.layers.Embedding(input_size, hidden_size, embeddings_initializer=tf.initializers.RandomNormal(0.0, 1.0))  # different: pad token embedding mask.
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                hidden_size,
                dropout=dropout,
                return_sequences=True,
                return_state=True))  # different: initializer, input shape.
        # self.gru2 = tf.keras.layers.Bidirectional(
        #    tf.keras.layers.RNN(
        #       tf.keras.layers.GRUCell(hidden_size,
        #                               kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size))),
        #                               recurrent_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size))),
        #                               bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size)))),
        #       return_sequences=True,
        #       return_state=True
        #    )
        #)
        self.gru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(
               tf.keras.layers.GRUCell(hidden_size,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size))),
                                       recurrent_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(hidden_size)),(1/np.sqrt(hidden_size)))),
               return_sequences=True,
               return_state=True
            )
        )
        # self.gru2 = tf.keras.layers.GRU(hidden_size, dropout=dropout, return_state=True, return_sequences=True)
        self.W = tf.keras.layers.Dense(hidden_size,
                                       use_bias=True,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*hidden_size)),(1/np.sqrt(2*hidden_size)))
                                       )  # different: bias should be explicitly assigned.

    def initialize_hidden_state(self, batch_size):
        forward_hidden = tf.zeros((batch_size, self.hidden_size))
        backward_hidden = tf.zeros((batch_size, self.hidden_size))
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

    def call(self, input_seqs, input_lengths, hidden=None, training=True):
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
        outputs, hidden_f, hidden_b = self.gru2(embedded,
                                                mask=mask,
                                                initial_state=hidden,
                                                training=training)  # need to check the meaning of outpus!!! not sure!!! different: padded token not mask in forward calculation, need a flag to indicate train or test if using dropout. No pack_padded_sequence and pad_packed_sequence.
        #pdb.set_trace()
        #outputs, hidden_f, cell_f, hidden_b, cell_b = self.gru2(embedded,
        #                                        mask=mask,
        #                                        training=training)  # need to check the meaning of outpus!!! not sure!!! different: padded token not mask in forward calculation, need a flag to indicate train or test if using dropout. No pack_padded_sequence and pad_packed_sequence.
        #hidden_hat = tf.concat([hidden_f, hidden_b], 1)
        hidden_hat = tf.concat([hidden_f, hidden_b], 1)
        hidden = self.W(hidden_hat)  # different: no unsqueeze(0).
        outputs = self.W(outputs)  # different: no need to transpose(0, 1) because the first dimension is already batch_size.
        return outputs, hidden




