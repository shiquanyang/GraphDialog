import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow_models.Libraries.GraphGRUCell import GraphGRUCell
from tensorflow.python.keras.utils import generic_utils
import numpy as np
import pdb


class RNN(tf.keras.Model):
    '''
    Base class for RNN layer.
    '''
    def __init__(self,
                 units,
                 input_dim,
                 edge_types,
                 shared_emb,
                 recurrent_size=4,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 unroll=True,
                 time_major=False,
                 zero_output_for_mask=True,
                 **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.edge_types = edge_types
        self.edge_embeddings = shared_emb
        self.recurrent_size = recurrent_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.unroll = unroll
        self.time_major = time_major
        self.zero_output_for_mask = zero_output_for_mask
        self.supports_masking = True
        self.cell = GraphGRUCell(units,
                                 input_dim,
                                 edge_types,
                                 shared_emb,
                                 recurrent_size,
                                 kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(units)),(1/np.sqrt(units))),
                                 # kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
                                 # kernel_initializer=tf.initializers.glorot_uniform,
                                 recurrent_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(units)),(1/np.sqrt(units))),
                                 # recurrent_initializer=tf.initializers.orthogonal,
                                 bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(units)),(1/np.sqrt(units))))
                                 # bias_initializer = tf.initializers.zeros)

    def call(self,
             inputs,  # inputs: batch_size*max_len*embedding_dim
             input_lengths,  # input_lengths: batch_size
             dependencies,  # dependencies: batch_size*max_len*recurrent_size
             edge_types,  # edge_types: batch_size*max_len*recurrent_size
             mask=None,  # mask: batch_size*max_len
             cell_mask=None,  # mask: batch_size*max_len*recurrent_size
             initial_states=None,  # initial_states: 4*batch_size*embedding_dim
             training=True):
        timesteps = inputs.shape[0] if self.time_major else inputs.shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the time dimension is undefined.')

        def step(inputs, states, edge_types, cell_mask, training):
            # pdb.set_trace()
            output, new_states = self.cell(inputs, states, edge_types, cell_mask, training)  # inputs: batch_size*embedding_dim, states: 4*batch_size*embedding_dim
            if not nest.is_sequence(new_states):
                new_states = [new_states]
            return output, new_states

        def swap_batch_timestep(input_t):
            axes = list(range(len(input_t.shape)))
            axes[0], axes[1] = 1, 0
            return array_ops.transpose(input_t, axes)

        if not self.time_major:
            inputs = nest.map_structure(swap_batch_timestep, inputs)  # inputs: max_len*batch_size*embedding_dim
            dependencies = swap_batch_timestep(dependencies)  # dependencies: max_len*batch_size*recurrent_size
            cell_mask = swap_batch_timestep(cell_mask)  # cell_mask: max_len*batch_size*recurrent_size
            edge_types = swap_batch_timestep(edge_types)  # edge_types: max_len*batch_size*recurrent_size

        flatted_inputs = nest.flatten(inputs)  # inputs: max_len*batch_size*embedding_dim
        time_steps = flatted_inputs[0].shape[0]
        batch = flatted_inputs[0].shape[1]

        for input_ in flatted_inputs:
            input_.shape.with_rank_at_least(3)

        if mask is not None:  # mask: batch_size*max_len
            if mask.dtype != dtypes_module.bool:
                mask = math_ops.cast(mask, dtypes_module.bool)
            if len(mask.shape) == 2:
                mask = array_ops.expand_dims(mask, axis=-1)  # mask: batch_size*max_len*1
            if not self.time_major:
                mask = swap_batch_timestep(mask)  # mask: max_len*batch_size*1

        if cell_mask is not None:
            if cell_mask.dtype != dtypes_module.bool:
                cell_mask = math_ops.cast(cell_mask, dtypes_module.bool)

        def _expand_mask(mask_t, input_t, fixed_dim=1):  # mask_t: batch_size*1, input_t: batch_size*embedding_dim
            assert not nest.is_sequence(mask_t)
            assert not nest.is_sequence(input_t)
            rank_diff = len(input_t.shape) - len(mask_t.shape)  # rand_diff: 0
            for _ in range(rank_diff):
                mask_t = array_ops.expand_dims(mask_t, -1)
            multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]  # multiples: [1, embedding_dim]
            return array_ops.tile(mask_t, multiples)

        if self.unroll:
            if not time_steps:
                raise ValueError('Unrolling requires a fixed number of timesteps.')
            states = tuple(initial_states)  # initial_states: 4*batch_size*embedding_dim
            successive_states = []
            successive_outputs = []

            def _process_single_input_t(input_t):
                input_t = array_ops.unstack(input_t)
                if self.go_backwards:
                    input_t.reverse()
                return input_t

            if nest.is_sequence(inputs):  # inputs: max_len*batch_size*embedding_dim
                processed_input = nest.map_structure(_process_single_input_t, inputs)
            else:
                processed_input = (_process_single_input_t(inputs),)

            def _get_input_tensor(time):
                inp = [t_[time] for t_ in processed_input]
                return nest.pack_sequence_as(inputs, inp)

            if mask is not None:
                mask_list = array_ops.unstack(mask)  # mask: max_len*batch_size*1
                if self.go_backwards:
                    mask = tf.reverse(mask, [0])
                    mask_list.reverse()

                # pdb.set_trace()
                for i in range(time_steps):
                    inp = _get_input_tensor(i)  # inp: batch_size*embedding_dim
                    mask_t = mask_list[i]  # mask_t: batch_size*1
                    if i < time_steps - 1:
                        dep_t = dependencies[i+1]  # dep_t: batch_size*recurrent_size
                    edge_types_t = edge_types[i]  # edge_types_t: batch_size*recurrent_size
                    cell_mask_t = cell_mask[i]  # cell_mask_t: batch_size*recurrent_size
                    output, new_states = step(inp, tuple(states), edge_types_t, cell_mask_t, training)  # inp: batch_size*embedding_dim, states: 4*batch_size*embedding_dim
                    # output: batch_size*embedding_dim, new_states:1*batch_size*embedding_dim
                    tiled_mask_t = _expand_mask(mask_t, output)  # tiled_mask_t: batch_size*embedding_dim

                    if not successive_outputs:
                        pre_output = array_ops.zeros_like(output)
                    else:
                        pre_output = successive_outputs[-1]

                    output = array_ops.where(tiled_mask_t, output, pre_output)  # output: batch_size*embedding_dim

                    # deal with masking
                    if not successive_states:
                        pre_states = array_ops.zeros_like(new_states)  # new_states: 1*batch_size*embedding_dim, pre_states: 1*batch_size*embedding_dim
                    else:
                        pre_states = successive_states[-1]

                    return_states = []
                    for state, new_state in zip(pre_states, new_states):
                        tiled_mask_t = _expand_mask(mask_t, new_state)
                        return_states.append(array_ops.where(tiled_mask_t, new_state, state))
                    # for state, new_state in zip(states, new_states):  # states: 4*batch_size*embedding_dim, new_states: 1*batch_size_embedding_dim
                    #     tiled_mask_t = _expand_mask(mask_t, new_state)
                    #     return_states.append(array_ops.where(tiled_mask_t, new_state, state))

                    successive_outputs.append(output)
                    successive_states.append(return_states)
                    # successive_states.append(states)

                    # get next timestep hidden input
                    # states[0] = return_states
                    if i < time_steps - 1:
                        states = []
                        states.append(return_states[0])
                        for k in range(self.recurrent_size - 1):
                            stack_t = []
                            for t in range(batch):
                                dep = dep_t[t, k]
                                if dep.numpy().decode() == '$':
                                    dep_state = array_ops.zeros([self.units])
                                else:
                                    if self.go_backwards:
                                        dep_state = successive_states[int(dep.numpy().decode())+(timesteps-input_lengths[t])][0][t]
                                    else:
                                        dep_state = successive_states[int(dep.numpy().decode())][0][t]
                                # states[k + 1, t] = dep_state
                                stack_t.append(dep_state)
                            stack_t = tf.stack(stack_t, axis=0)
                            states.append(stack_t)

                # pdb.set_trace()
                last_output = successive_outputs[-1]  # last_output: batch_size*embedding_dim
                new_states = successive_states[-1]  # new_states: batch_size*embedding_dim
                outputs = array_ops.stack(successive_outputs)  # outputs: max_len*batch_size*embedding_dim

                if self.zero_output_for_mask:
                    last_output = array_ops.where(
                        _expand_mask(mask_list[-1], last_output),  # mask_list[-1]: batch_size*1, last_output: batch_size*embedding_dim
                        last_output,  # last_output: batch_size*embedding_dim
                        array_ops.zeros_like(last_output))
                    outputs = array_ops.where(
                        _expand_mask(mask, outputs, fixed_dim=2),  # mask: max_len*batch_size*1, outputs: max_len*batch_size*embedding_dim
                        outputs,  # outputs: max_len*batch_size*embedding_dim
                        array_ops.zeros_like(outputs))

        def set_shape(output_):
            if isinstance(output_, ops.Tensor):
                shape = output_.shape.as_list()
                shape[0] = time_steps
                shape[1] = batch
                output_.set_shape(shape)
            return output_

        outputs = nest.map_structure(set_shape, outputs)

        if not self.time_major:
            outputs = nest.map_structure(swap_batch_timestep, outputs)  # outputs: batch_size*max_len*embedding_dim

        # return last_output, outputs, new_states
        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(new_states, (list, tuple)):
                states = [new_states]
            else:
                states = list(new_states)
            return generic_utils.to_list(output) + states
        else:
            return output