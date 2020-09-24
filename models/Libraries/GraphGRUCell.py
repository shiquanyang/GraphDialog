import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import math_ops
from utils.config import *
import numpy as np
import pdb


class GraphGRUCell(tf.keras.Model):
    '''
    Cell class for GraphGRU layer.
    '''
    def __init__(self,
                 units,
                 input_dim,
                 edge_types,
                 shared_emb,
                 recurrent_size=4,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GraphGRUCell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.edge_types = edge_types
        self.edge_embeddings = shared_emb
        self.recurrent_size = recurrent_size

        self.activation = tf.keras.layers.Activation(activation)
        self.recurrent_activation = tf.keras.layers.Activation(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.softmax = tf.keras.layers.Softmax(1)

        self.kernel = self.add_weight(  # self.kernel: input_dim*(3*embedding_dim)
            name='kernel',
            shape=(input_dim, 3 * units),
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            constraint=kernel_constraint
        )

        # create sharing kernels for all edge types
        self.recurrent_kernel = self.add_weight(  # self.recurrent_kernel: recurrent_size*embedding_dim*(3*embedding_dim)
            name='recurrent_kernel',
            shape=(recurrent_size, units, 3 * units),
            initializer=recurrent_initializer,
            regularizer=recurrent_regularizer,
            constraint=recurrent_constraint
        )

        # create kernels for each edge type
        # self.recurrent_kernel = self.add_weight(  # self.recurrent_kernel: recurrent_size*embedding_dim*(3*embedding_dim)
        #     name='recurrent_kernel',
        #     shape=(edge_types, units, 3 * units),
        #     initializer=recurrent_initializer,
        #     regularizer=recurrent_regularizer,
        #     constraint=recurrent_constraint
        # )

        # create sharing biases for all edge types
        if use_bias:
            self.bias = self.add_weight(  # self.bias: (recurrent_size+1)*(3*embedding_dim)
                name='bias',
                shape=((recurrent_size + 1), 3 * units),
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint
            )
        else:
            self.bias = None

        # create biases for each edge type
        # if use_bias:
        #     self.bias = self.add_weight(  # self.bias: (recurrent_size+1)*(3*embedding_dim)
        #         name='bias',
        #         shape=((edge_types + 1), 3 * units),
        #         initializer=bias_initializer,
        #         regularizer=bias_regularizer,
        #         constraint=bias_constraint
        #     )
        # else:
        #     self.bias = None

        # add for additive attention
        self.v = self.add_weight(
            name='v',
            shape=(self.units, 1),
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            constraint=bias_constraint
        )

    def call(self, inputs, states, edge_types, cell_mask, training=True):  # inputs: batch_size*embedding_dim, states:4*batch_size*embedding_dim, cell_mask: batch_size*recurrent_size
        batch_size = inputs.shape[0]
        state_size = len(states)
        if state_size > self.recurrent_size:
            raise ValueError("length of states exceeds recurrent_size.")
        if self.use_bias:
            unstacked_biases = array_ops.unstack(self.bias)  # unstacked_biases: (recurrent_size+1)*embedding_dim
            input_bias, recurrent_bias = unstacked_biases[0], unstacked_biases[1:]  # input_bias: (3*embedding_dim), recurrent_bias: recurrent_size*(3*embedding_dim)

        matrix_x = K.dot(inputs, self.kernel)  # matrix_x: batch_size*(3*embedding_dim)
        if self.use_bias:
            # biases: bias_z_i, bias_r_i, bias_h_i
            matrix_x = K.bias_add(matrix_x, input_bias)

        x_z = matrix_x[:, :self.units]  # x_z: batch_size*embedding_dim
        x_r = matrix_x[:, self.units: 2 * self.units]  # x_r: batch_size*embedding_dim
        x_h = matrix_x[:, 2 * self.units:]  # x_h: batch_size*embedding_dim

        def _expand_mask(mask_t, input_t, fixed_dim=1):  # mask_t: batch_size*1, input_t: batch_size*embedding_dim
            assert not nest.is_sequence(mask_t)
            assert not nest.is_sequence(input_t)
            rank_diff = len(input_t.shape) - len(mask_t.shape)  # rand_diff: 0
            for _ in range(rank_diff):
                mask_t = array_ops.expand_dims(mask_t, -1)
            multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]  # multiples: [1, embedding_dim]
            return array_ops.tile(mask_t, multiples)
        # comment for sum_after
        accumulate_h = array_ops.zeros([batch_size, self.units])  # accumulate_h: batch_size*embedding_dim
        accumulate_z_h = array_ops.zeros([batch_size, self.units])  # accumulate_z_h: batch_size*embedding_dim
        accumulate_z = array_ops.zeros([batch_size, self.units])  # accumulate_z: batch_size*embedding_dim

        # add for sum_after
        # h_hat = []
        loop = 1 if args['ablationD'] else self.recurrent_size

        z_list = []
        h_list = []
        for k in range(loop):
            # edge embedding
            edge_embed = self.edge_embeddings(edge_types[:, k])  # edge_embed: batch_size*embedding
            # mask
            tiled_mask_t = _expand_mask(cell_mask[:, k], edge_embed)  # tiled_mask_t: batch_size*embedding_dim
            edge_embed = array_ops.where(tiled_mask_t, edge_embed, array_ops.ones_like(edge_embed))  # edge_embed: batch_size*embedding_dim
            # state = states[k] * edge_embed  # state: batch_size*embedding_dim
            # add for mimic baseline
            state = states[k]  # state: batch_size*embedding_dim
            h_list.append(state)

            # gather recurrent kernels and biases according input edge types
            # matrix_inner = []
            # for t in range(batch_size):
            #     edge_type = edge_types[t, k]
            #     kernel = self.recurrent_kernel[edge_type]
            #     bias = recurrent_bias[edge_type]
            #     matrix_inner_t = K.dot(tf.expand_dims(state[t], axis=0), kernel)
            #     if self.use_bias:
            #         matrix_inner_t = K.bias_add(matrix_inner_t, bias)
            #     matrix_inner.append(tf.squeeze(matrix_inner_t, axis=0))
            # matrix_inner = tf.stack(matrix_inner, axis=0)

            matrix_inner = K.dot(state, self.recurrent_kernel[k])  # matrix_inner: batch_size*(3*embedding_dim), states[k]: batch_size*embedding_dim
            if self.use_bias:
                matrix_inner = K.bias_add(matrix_inner, recurrent_bias[k])

            recurrent_z = matrix_inner[:, :self.units]  # recurrent_z: batch_size*embedding_dim
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]  # recurrent_r: batch_size*embedding_dim

            # add for softmax attention
            z_list.append(recurrent_z)

            z = self.recurrent_activation(x_z + recurrent_z)  # z: batch_size*embedding_dim
            r = self.recurrent_activation(x_r + recurrent_r)  # r: batch_size*embedding_dim

            # add for sum_after
            # hh = self.activation(x_h + r * matrix_inner[:, 2 * self.units:])
            # h = (1 - z) * hh + z * state
            # h = array_ops.where(tiled_mask_t, h, array_ops.zeros_like(h))
            # h_hat.append(h)

            # comment for sum_after
            recurrent_h = r * matrix_inner[:, 2 * self.units:]  # recurrent_h: batch_size*embedding_dim
            recurrent_h = array_ops.where(tiled_mask_t, recurrent_h, array_ops.zeros_like(recurrent_h))  # recurrent_h: batch_size*embedding_dim
            accumulate_h = accumulate_h + recurrent_h  # accumulate_h: batch_size*embedding_dim

            # comment for softmax attention
            # z_h = z * state
            # z_h = array_ops.where(tiled_mask_t, z_h, array_ops.zeros_like(z_h))
            # accumulate_z_h = accumulate_z_h + z_h  # accumulate_z_h: batch_size*embedding_dim

            # comment for softmax attention
            # z = array_ops.where(tiled_mask_t, z, array_ops.zeros_like(z))
            # accumulate_z = accumulate_z + z  # accumulate_z: batch_size*embedding_dim

        # add for sum_after
        # if args['ablationD']:
        #     o_h = h_hat[0]
        # else:
        #     h_hat = tf.reduce_sum(tf.stack(h_hat, axis=0), axis=0)
        #     cell_mask = tf.reduce_sum(math_ops.cast(cell_mask, dtypes_module.int32), axis=1)
        #     o_h = h_hat / tf.cast(tf.tile(tf.expand_dims(cell_mask, axis=1), [1, self.units]), dtype=float)

        # commet for sum_after
        # actual_num = tf.cast(tf.reduce_sum(math_ops.cast(cell_mask, dtypes_module.int32), axis=1, keepdims=True), dtype=float)  # actual_num: batch_size
        hh = self.activation(x_h + accumulate_h / loop)  # hh: batch_size*embedding_dim
        h_list.append(hh)  # h_list: input_hidden without linear
        z_list.append(hh)  # z_list: input_hidden after linear

        # add for softmax attention
        hidden_bank = tf.transpose(tf.stack(z_list, axis=0), [1, 0, 2])  # hidden_memory: batch_size * (recurrent_size + 1) * embedding_dim
        x_z_temp = tf.tile(tf.expand_dims(x_z, axis=1), [1, hidden_bank.shape[1], 1])  # x_z_temp = batch_size * (recurrent_size + 1) * embedding_dim
        # add for additive attention
        prob_logits = tf.matmul(tf.tile(tf.expand_dims(tf.transpose(self.v, [1, 0]), axis=0), [batch_size, 1, 1]), tf.transpose(self.activation(x_z_temp + hidden_bank), [0, 2, 1]))
        prob_logits = tf.squeeze(tf.transpose(prob_logits, [0, 2, 1]), axis=2)
        # comment for additive attention
        # prob_logits = tf.reduce_sum(hidden_bank * x_z_temp, axis=2)  # prob_logits: batch_size * (recurrent_size + 1)
        # add for masked softmax
        mask_list = []
        cell_mask_slices = tf.split(cell_mask, num_or_size_splits=cell_mask.shape[1], axis=1)
        for tensor in cell_mask_slices:
            mask_list.append(tensor)
        hh_mask = tf.ones([batch_size, 1], dtype=tf.bool)  # hh_mask: batch_size * 1
        mask_list.append(hh_mask)
        new_mask = tf.squeeze(tf.stack(mask_list, axis=1), axis=2)  # new_mask: batch_size * (recurrent_size + 1)
        prob_logits_temp = array_ops.where(new_mask, prob_logits, (-1 * np.ones_like(prob_logits) * np.inf))
        prob_soft = self.softmax(prob_logits_temp)  # prob_soft: batch_size * (recurrent_size + 1)
        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, axis=2), [1, 1, hidden_bank.shape[2]])  # prob_soft_temp: batch_size * (recurrent_size + 1) * embedding_dim
        output_hidden_bank = tf.transpose(tf.stack(h_list, axis=0), [1, 0, 2])  # output_hidden_bank: batch_size * (recurrent_size + 1) * embedding_dim
        h = tf.reduce_sum((output_hidden_bank * prob_soft_temp), axis=1)  # h: batch_size * embedding_dim

        # comment for softmax attention
        # h = (1 - accumulate_z / loop) * hh + accumulate_z_h / loop  # h: batch_size*embedding_dim
        return h, [h]

        # add for sum_after
        # return o_h, [o_h]