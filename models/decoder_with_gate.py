import tensorflow as tf
from utils.config import *
import numpy as np
import pdb
from tensorflow.python.ops import embedding_ops


class LocalMemoryDecoder(tf.keras.Model):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.C = shared_emb
        self.softmax = tf.keras.layers.Softmax(1)
        self.sketch_rnn = tf.keras.layers.GRU(embedding_dim,
                                              dropout=dropout,
                                              return_sequences=True,
                                              return_state=True)  # different: need to set training flag if using dropout.
        self.sketch_rnn2 = tf.keras.layers.RNN(
               tf.keras.layers.GRUCell(embedding_dim,
                                       kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(embedding_dim)),(1/np.sqrt(embedding_dim))),
                                       recurrent_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(embedding_dim)),(1/np.sqrt(embedding_dim))),
                                       bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(embedding_dim)),(1/np.sqrt(embedding_dim)))),
               return_sequences=True,
               return_state=True
            )
        self.relu = tf.keras.layers.ReLU()
        self.projector = tf.keras.layers.Dense(embedding_dim,
                                               use_bias=True,
                                               kernel_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*embedding_dim)),(1/np.sqrt(2*embedding_dim))),
                                               bias_initializer=tf.initializers.RandomUniform(-(1/np.sqrt(2*embedding_dim)),(1/np.sqrt(2*embedding_dim)))
                                               )
        self.gate_layer1 = tf.keras.layers.Dense(2 * embedding_dim,
                                                 use_bias=True,
                                                 kernel_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(4 * embedding_dim)), (1 / np.sqrt(4 * embedding_dim))),
                                                 bias_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(4 * embedding_dim)), (1 / np.sqrt(4 * embedding_dim)))
                                                 )
        self.gate_layer2 = tf.keras.layers.Dense(3,
                                                 use_bias=True,
                                                 kernel_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(6)), (1 / np.sqrt(6))),
                                                 bias_initializer=tf.initializers.RandomUniform(-(1 / np.sqrt(6)), (1 / np.sqrt(6)))
                                                 )
        self.softmax = tf.keras.layers.Softmax(1)

    def attend_vocab(self, seq, cond):
        seq[PAD_token, :] = np.zeros([1, self.embedding_dim], dtype=np.float32)
        scores_ = tf.matmul(cond, tf.transpose(seq))  # different: no softmax layer, need to check loss function.
        return scores_

    def gen_embedding_mask(self, input):
        raw_mask_array = [[1.0]] * PAD_token + [[0.0]] + [[1.0]] * (self.num_vocab - PAD_token - 1)
        mask = embedding_ops.embedding_lookup(raw_mask_array, tf.cast(input, dtype=tf.int32))
        ret_mask = tf.tile(tf.expand_dims(mask, 1), [1, self.embedding_dim])
        return ret_mask

    def call(self, extKnow, story_size, story_lengths, copy_list, encode_hidden,
             target_batches, max_target_length, batch_size, use_teacher_forcing,
             get_decoded_words, global_pointer, kb_len, conv_len, training=True):
        # pdb.set_trace()
        # all_decoder_outputs_vocab = tf.zeros([max_target_length.numpy()[0], batch_size, self.num_vocab])  # max_target_length * batch_size * num_vocab.
        # all_decoder_outputs_ptr = tf.zeros([max_target_length.numpy()[0], batch_size, story_size[1]])  # max_target_length * batch_size * memory_size.
        # memory_mask_for_step = tf.ones([story_size[0], story_size[1]])  # batch_size * memory_size.
        memory_mask_for_step = np.ones((story_size[0], story_size[1]), dtype=np.float32)  # batch_size * memory_size.
        decoded_fine, decoded_coarse = [], []

        decoder_input = tf.constant([SOS_token] * batch_size)  # batch_size.
        hidden = self.relu(self.projector(encode_hidden))  # batch_size * embedding_dim.
        # hidden = encode_hidden  # batch_size * embedding_dim.

        all_decoder_outputs_vocab = []
        all_decoder_outputs_ptr = []
        all_decoder_outputs_gate_signal = []
        for t in range(max_target_length):
            embed_q = self.C(decoder_input)
            if len(decoder_input.shape) == 0:  # for batch_size = 1
                decoder_input = tf.expand_dims(decoder_input, 0)
                embed_q = tf.expand_dims(embed_q, 0)
            pad_mask = self.gen_embedding_mask(decoder_input)
            embed_q = tf.multiply(embed_q, pad_mask)
            if training:
                embed_q = self.dropout_layer(embed_q, training=training)  # batch_size * embedding_dim.
            if len(embed_q.get_shape()) == 1:
                embed_q = tf.expand_dims(embed_q, 0)
            _, hidden = self.sketch_rnn2(tf.expand_dims(embed_q, 1),
                                         initial_state=hidden,
                                         training=training)  # 1 * batch_size * embedding_dim.
            query_vector = hidden  # need to check meaning of hidden[0], query_vector: batch_size * embedding_dim.
            p_vocab = self.attend_vocab(self.C.embeddings.numpy(), hidden)  # self.C.read_value: num_vocab * embedding_dim, p_vocab: batch_size * num_vocab.
            # all_decoder_outputs_vocab[t] = p_vocab
            all_decoder_outputs_vocab.append(p_vocab)
            _, topvi = tf.math.top_k(p_vocab)  # topvi: batch_size * 1.

            gate_signal = self.relu(self.gate_layer1(query_vector))
            if training:
                gate_signal = self.dropout_layer(gate_signal, training=training)
            gate_signal = self.gate_layer2(gate_signal)
            gate_signal = self.softmax(gate_signal)
            all_decoder_outputs_gate_signal.append(gate_signal)

            prob_soft, prob_logits = extKnow(query_vector, global_pointer, gate_signal, kb_len, conv_len, training=training)  # query_vector: batch_size * embedding_dim, global_pointer: batch_size * memory_size.
            # all_decoder_outputs_ptr[t] = prob_logits  # need to check whether use softmax or not, prob_logits: batch_size * memory_size.
            all_decoder_outputs_ptr.append(prob_logits)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]  # decoder_input: batch_size, target_batches[:, t].
            else:
                decoder_input = tf.squeeze(topvi)  # decoder_input: batch_size.

            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = tf.math.top_k(prob_soft, k=search_len)  # toppi: batch_size * search_len.
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi, 0].numpy().astype(int)
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[bi, i].numpy().astype(int) < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[bi, i].numpy().astype(int)]
                                break
                        temp_f.append(cw)
                        if args['record']:
                            memory_mask_for_step[bi, toppi[bi, i].numpy().astype(int)] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])
                decoded_coarse.append(temp_c)
                decoded_fine.append(temp_f)
        all_decoder_outputs_vocab_out = tf.stack(all_decoder_outputs_vocab, axis=0)
        all_decoder_outputs_ptr_out = tf.stack(all_decoder_outputs_ptr, axis=0)
        return all_decoder_outputs_vocab_out, all_decoder_outputs_ptr_out, decoded_fine, decoded_coarse, all_decoder_outputs_gate_signal