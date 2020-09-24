import tensorflow as tf
import numpy as np
from utils.config import *


class Dataset:
    def __init__(self, data_info, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data_info = {}
        self.data_size = len(data_info)
        for key in data_info[0].keys():  # information organization form change from sample-wise to information_type-wise.
            self.data_info[key] = []
        for elm in data_info:
            for key in elm:
                self.data_info[key].append(elm[key])  # already id, but not padding.
        self.shuffle = shuffle
        self.on_epoch_end()

    def len(self):
        return int(np.floor(self.data_size) / self.batch_size)

    def get_batch(self, index, is_last=False):
        if not is_last:
            shuffled_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            shuffled_indexes = self.indexes[self.len()*self.batch_size:]
        batch_data = self.data_generation(shuffled_indexes)
        padded_data = self.batch_padding(batch_data)
        return padded_data

    def load_batches(self, drop_last=True):
        batches = []
        for index in range(self.len()):
            batches.append(self.get_batch(index, False))
        if not drop_last and self.data_size % self.batch_size != 0:
            batches.append(self.get_batch(1e-5, True))
        return batches

    def padding(self, sequences, story_dim):
        '''
        Pad word sequences.
        :param sequences:
        :param story_dim:
        :return:
        '''
        # lengths = [[len(seq)] for seq in sequences]
        lengths_int = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths_int) == 0 else max(lengths_int)
        if (story_dim):
            # padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
            # padded_seqs = tf.ones([len(sequences), max_len, MEM_TOKEN_SIZE], dtype=tf.dtypes.int64)
            padded_seqs = np.ones([len(sequences), max_len, MEM_TOKEN_SIZE], dtype=np.dtype(np.int64))
            for i, seq in enumerate(sequences):
                end = lengths_int[i]
                if len(seq) != 0:
                    padded_seqs[i, :end, :] = seq[:end]
        else:
            # padded_seqs = torch.ones(len(sequences), max_len).long()
            # padded_seqs = tf.ones([len(sequences), max_len], dtype=tf.dtypes.int64)
            padded_seqs = np.ones([len(sequences), max_len], dtype=np.dtype(np.int64))
            for i, seq in enumerate(sequences):
                end = lengths_int[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths_int

    def padding_index(self, sequences):
        '''
        Pad non-word sequences.
        :param sequences:
        :return:
        '''
        # lengths = [[len(seq)] for seq in sequences]
        lengths_int = [len(seq) for seq in sequences]
        # padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
        # padded_seqs = tf.zeros([len(sequences), max(lengths)], dtype=tf.dtypes.float32)
        padded_seqs = np.zeros([len(sequences), max(lengths_int)], dtype=np.dtype(np.float))
        for i, seq in enumerate(sequences):
            end = lengths_int[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths_int

    def batch_padding(self, data):
        '''
        Padding the data according data category.
        :param data:
        :return:
        '''
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]
        context_arr, context_arr_lengths = self.padding(item_info['context_arr'], True)
        response, response_lengths = self.padding(item_info['response'], False)
        selector_index, _ = self.padding_index(item_info['selector_index'])
        ptr_index, _ = self.padding(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = self.padding(item_info['conv_arr'], True)
        sketch_response, _ = self.padding(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = self.padding(item_info['kb_arr'], True)

        data_info = {}
        for key in item_info.keys():
            try:
                data_info[key] = locals()[key]
            except:
                data_info[key] = item_info[key]

        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        return data_info

    def data_generation(self, indexes):
        '''
        Generate a batch of data according to the indexes of a batch.
        :param indexes:
        :return:
        '''
        data = []
        for id in indexes:
            temp_info = {}
            for key in self.data_info:
                temp_info[key] = self.data_info[key][id]
            data.append(temp_info)
        return data

    def on_epoch_end(self):
        self.indexes = np.arange(self.data_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

