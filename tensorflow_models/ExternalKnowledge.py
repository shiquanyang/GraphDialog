import tensorflow as tf
from utils.config import *
import pdb
from tensorflow.python.ops import embedding_ops
from collections import OrderedDict


class ExternalKnowledge(tf.keras.Model):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.dropout = dropout
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        # 3-hops
        #self.module_list = OrderedDict()
        #for hop in range(self.max_hops+1):
        #    C = tf.keras.layers.Embedding(self.vocab,
        #                                  self.embedding_dim,
        #                                  embeddings_initializer=tf.initializers.RandomNormal(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        #    self.module_list['C_{}'.format(hop)] = C
        self.C_1 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.initializers.RandomNormal(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.C_2 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.initializers.RandomNormal(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.C_3 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.initializers.RandomNormal(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.C_4 = tf.keras.layers.Embedding(self.vocab,
                                          self.embedding_dim,
                                          embeddings_initializer=tf.initializers.RandomNormal(0.0, 0.1))  # different: no masking for pad token, pad token embedding does not equal zero, only support one hop.
        self.softmax = tf.keras.layers.Softmax(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        output_list = []
        for bi in range(full_memory.shape[0]):
            stack_list = []
            kb_aligns = tf.zeros([kb_len[bi]-1, full_memory.shape[2]])
            null_aligns = tf.zeros([1, full_memory.shape[2]])
            # print('kb_len:', kb_len[bi].numpy())
            if (kb_len[bi]-1).numpy().item() != 0:
                kb_aligns_slices = tf.split(kb_aligns, num_or_size_splits=kb_aligns.shape[0], axis=0)
                for tensor in kb_aligns_slices:
                    stack_list.append(tensor)
            # pdb.set_trace()
            hiddens_slices = tf.split(hiddens[bi, :conv_len[bi], :], num_or_size_splits=conv_len[bi].numpy().item(), axis=0)
            # print('hiddens len:', hiddens[bi].shape[0])
            for tensor in hiddens_slices:
                stack_list.append(tensor)
            # pdb.set_trace()
            stack_list.append(null_aligns)
            pad_len = full_memory.shape[1] - kb_len[bi] - conv_len[bi]
            # pdb.set_trace()
            if pad_len.numpy().item() != 0:
                pad_aligns = tf.zeros([pad_len, full_memory.shape[2]])
                pad_aligns_slice = tf.split(pad_aligns, num_or_size_splits=pad_aligns.shape[0], axis=0)
                for tensor in pad_aligns_slice:
                    stack_list.append(tensor)
                # pdb.set_trace()
            # print('stack_list length:', len(stack_list))
            add_tensor = tf.squeeze(tf.stack(stack_list, axis=0), 1)
            updated_full_memory = full_memory[bi] + add_tensor
            output_list.append(updated_full_memory)
        output = tf.stack(output_list, axis=0)
        return output

    def gen_embedding_mask(self, input):
        raw_mask_array = [[1.0]] * PAD_token + [[0.0]] + [[1.0]] * (self.vocab - PAD_token - 1)
        mask = embedding_ops.embedding_lookup(raw_mask_array, tf.cast(input, dtype=tf.int32))
        ret_mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, self.embedding_dim])
        return ret_mask

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, training=True):
        u = [hidden]  # different: hidden without squeeze(0), hidden: batch_size * embedding_size.
        story_size = story.shape
        self.m_story = []
        
        # hop-1
        #for hop in range(self.max_hops):
        embedding_A = self.C_1(tf.reshape(story, [story_size[0], -1]))  # story: batch_size * seq_len * MEM_TOKEN_SIZE, embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_A = tf.multiply(embedding_A, pad_mask)
        embedding_A = tf.reshape(embedding_A, [story_size[0], story_size[1], story_size[2], embedding_A.shape[-1]])  # embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        embedding_A = tf.math.reduce_sum(embedding_A, 2)  # embedding_A: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_A = self.add_lm_embedding(embedding_A, kb_len, conv_len, dh_outputs)
        if training:
            embedding_A = self.dropout_layer(embedding_A, training=training)

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embedding_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embedding_A * u_temp), 2)  # prob_logits: batch_size * memory_size
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size

        embedding_C = self.C_2(tf.reshape(story, [story_size[0], -1]))
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_C = tf.multiply(embedding_C, pad_mask)
        embedding_C = tf.reshape(embedding_C, [story_size[0], story_size[1], story_size[2], embedding_C.shape[-1]])
        embedding_C = tf.math.reduce_sum(embedding_C, 2)  # embedding_C: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_C = self.add_lm_embedding(embedding_C, kb_len, conv_len, dh_outputs)

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embedding_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embedding_C * prob_soft_temp), 1)
        u.append(u_k)
        self.m_story.append(embedding_A)
        
        # hop-2
        #for hop in range(self.max_hops):
        embedding_A = self.C_2(tf.reshape(story, [story_size[0], -1]))  # story: batch_size * seq_len * MEM_TOKEN_SIZE, embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_A = tf.multiply(embedding_A, pad_mask)
        embedding_A = tf.reshape(embedding_A, [story_size[0], story_size[1], story_size[2], embedding_A.shape[-1]])  # embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        embedding_A = tf.math.reduce_sum(embedding_A, 2)  # embedding_A: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_A = self.add_lm_embedding(embedding_A, kb_len, conv_len, dh_outputs)
        if training:
            embedding_A = self.dropout_layer(embedding_A, training=training)

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embedding_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embedding_A * u_temp), 2)  # prob_logits: batch_size * memory_size
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size

        embedding_C = self.C_3(tf.reshape(story, [story_size[0], -1]))
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_C = tf.multiply(embedding_C, pad_mask)
        embedding_C = tf.reshape(embedding_C, [story_size[0], story_size[1], story_size[2], embedding_C.shape[-1]])
        embedding_C = tf.math.reduce_sum(embedding_C, 2)  # embedding_C: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_C = self.add_lm_embedding(embedding_C, kb_len, conv_len, dh_outputs)

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embedding_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embedding_C * prob_soft_temp), 1)
        u.append(u_k)
        self.m_story.append(embedding_A)
        
        # hop-3
        #for hop in range(self.max_hops):
        embedding_A = self.C_3(tf.reshape(story, [story_size[0], -1]))  # story: batch_size * seq_len * MEM_TOKEN_SIZE, embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_A = tf.multiply(embedding_A, pad_mask)
        embedding_A = tf.reshape(embedding_A, [story_size[0], story_size[1], story_size[2], embedding_A.shape[-1]])  # embedding_A: batch_size * memory_size * MEM_TOKEN_SIZE * embedding_dim.
        embedding_A = tf.math.reduce_sum(embedding_A, 2)  # embedding_A: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_A = self.add_lm_embedding(embedding_A, kb_len, conv_len, dh_outputs)
        if training:
            embedding_A = self.dropout_layer(embedding_A, training=training)

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embedding_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embedding_A * u_temp), 2)  # prob_logits: batch_size * memory_size
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size

        embedding_C = self.C_4(tf.reshape(story, [story_size[0], -1]))
        pad_mask = self.gen_embedding_mask(tf.reshape(story, [story_size[0], -1]))
        # embedding_C = tf.multiply(embedding_C, pad_mask)
        embedding_C = tf.reshape(embedding_C, [story_size[0], story_size[1], story_size[2], embedding_C.shape[-1]])
        embedding_C = tf.math.reduce_sum(embedding_C, 2)  # embedding_C: batch_size * memory_size * embedding_dim.
        if not args['ablationH']:
            embedding_C = self.add_lm_embedding(embedding_C, kb_len, conv_len, dh_outputs)

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embedding_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embedding_C * prob_soft_temp), 1)
        u.append(u_k)
        self.m_story.append(embedding_A)
        
        self.m_story.append(embedding_C)

        return self.sigmoid(prob_logits), u[-1], prob_logits

    def call(self, query_vector, global_pointer, training=True):
        u = [query_vector]  # query_vector: batch_size * embedding_dim.

        # hop-1
        #for hop in range(self.max_hops):
        embed_A = self.m_story[0]  # embed_A: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_A = embed_A * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_A.shape[2]])

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embed_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embed_A * u_temp), 2)  # prob_logits: batch_size * memory_size.
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size.

        embed_C = self.m_story[1]  # embed_C: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_C = embed_C * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_C.shape[2]])

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embed_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embed_C * prob_soft_temp), 1)  # u_k: batch_size * embedding_dim.
        u.append(u_k)

        # hop-2
        #for hop in range(self.max_hops):
        embed_A = self.m_story[1]  # embed_A: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_A = embed_A * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_A.shape[2]])

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embed_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embed_A * u_temp), 2)  # prob_logits: batch_size * memory_size.
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size.

        embed_C = self.m_story[2]  # embed_C: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_C = embed_C * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_C.shape[2]])

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embed_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embed_C * prob_soft_temp), 1)  # u_k: batch_size * embedding_dim.
        u.append(u_k)

        # hop-3
        #for hop in range(self.max_hops):
        embed_A = self.m_story[2]  # embed_A: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_A = embed_A * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_A.shape[2]])

        u_temp = tf.tile(tf.expand_dims(u[-1], 1), [1, embed_A.shape[1], 1])  # u_temp: batch_size * memory_size * embedding_dim.
        prob_logits = tf.math.reduce_sum((embed_A * u_temp), 2)  # prob_logits: batch_size * memory_size.
        prob_soft = self.softmax(prob_logits)  # prob_soft: batch_size * memory_size.

        embed_C = self.m_story[3]  # embed_C: batch_size * memory_size * embedding_dim.
        if not args['ablationG']:
            embed_C = embed_C * tf.tile(tf.expand_dims(global_pointer, 2), [1, 1, embed_C.shape[2]])

        prob_soft_temp = tf.tile(tf.expand_dims(prob_soft, 2), [1, 1, embed_C.shape[2]])  # prob_soft_temp: batch_size * memory_size * embedding_dim.
        u_k = u[-1] + tf.math.reduce_sum((embed_C * prob_soft_temp), 1)   # u_k: batch_size * embedding_dim.
        u.append(u_k)

        return prob_soft, prob_logits


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
