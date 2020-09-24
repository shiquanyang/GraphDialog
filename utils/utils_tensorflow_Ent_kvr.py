import json
import ast
import tensorflow as tf
from utils.utils_general import *
import numpy as np
from utils.tensorflow_dataset import *
from utils.utils_tensorflow_generator_kvr import *
import pdb


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0

    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = gold_ent
                    elif task_type == "schedule":
                        ent_idx_cal = gold_ent
                    elif task_type == "navigate":
                        ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    a = 0
                    b = 0
                    for key in r.split():
                        a += 1
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key != 'poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = [sent_token[-1]]
        # sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def build_lang(pairs, type):
    '''
    Build vocabulary to index words.
    :param pairs:
    :param batch_size:
    :param type:
    :return:
    '''
    lang = Lang()
    for pair in pairs:
        if(type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)
    return lang


def preprocess(sequence, word2id, trg=True):
    """
    Converts words to ids.
    """
    if trg:
        story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
    else:
        story = []
        for i, word_triple in enumerate(sequence):
            story.append([])
            for ii, word in enumerate(word_triple):
                temp = word2id[word] if word in word2id else UNK_token
                story[i].append(temp)
    story = tf.convert_to_tensor(story)
    return story


def text_to_sequence(pairs, lang):
    '''
    Map texts to ids.
    :param pairs:
    :param lang:
    :return:
    '''
    sequence_data = []
    for pair in pairs:
        response_plain = []
        context_arr = preprocess(pair['context_arr'], lang.word2index, trg=False)
        response = preprocess(pair['response'], lang.word2index)
        ptr_index = tf.convert_to_tensor(pair['ptr_index'])
        selector_index = tf.convert_to_tensor(pair['selector_index'])
        conv_arr = preprocess(pair['conv_arr'], lang.word2index, trg=False)
        kb_arr = preprocess(pair['kb_arr'], lang.word2index, trg=False)
        sketch_response = preprocess(pair['sketch_response'], lang.word2index)
        # additional plain information
        context_arr_plain = pair['context_arr']
        response_plain.append(pair['response'])
        kb_arr_plain = pair['kb_arr']
        sequence_data.append({
            'context_arr':context_arr,
            'response':response,
            'ptr_index':ptr_index,
            'selector_index':selector_index,
            'conv_arr':conv_arr,
            'kb_arr':kb_arr,
            'sketch_response':sketch_response,
            'context_arr_plain':list(context_arr_plain),
            'response_plain':list(response_plain),
            'kb_arr_plain':list(kb_arr_plain),
            'ent_index':pair['ent_index'],
            'ent_idx_cal':pair['ent_idx_cal'],
            'ent_idx_nav':pair['ent_idx_nav'],
            'ent_idx_wet':pair['ent_idx_wet'],
            'ID':pair['ID']
        })
    return sequence_data


# def padding(sequences, story_dim):
#     '''
#     Pad word sequences.
#     :param sequences:
#     :param story_dim:
#     :return:
#     '''
#     lengths = [[len(seq)] for seq in sequences]
#     lengths_int = [len(seq) for seq in sequences]
#     max_len = 1 if max(lengths_int) == 0 else max(lengths_int)
#     if (story_dim):
#         # padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
#         # padded_seqs = tf.ones([len(sequences), max_len, MEM_TOKEN_SIZE], dtype=tf.dtypes.int64)
#         padded_seqs = np.ones([len(sequences), max_len, MEM_TOKEN_SIZE], dtype=np.dtype(np.int64))
#         for i, seq in enumerate(sequences):
#             end = lengths_int[i]
#             if len(seq) != 0:
#                 padded_seqs[i, :end, :] = seq[:end]
#     else:
#         # padded_seqs = torch.ones(len(sequences), max_len).long()
#         # padded_seqs = tf.ones([len(sequences), max_len], dtype=tf.dtypes.int64)
#         padded_seqs = np.ones([len(sequences), max_len], dtype=np.dtype(np.int64))
#         for i, seq in enumerate(sequences):
#             end = lengths_int[i]
#             padded_seqs[i, :end] = seq[:end]
#     return padded_seqs, lengths


# def padding_index(sequences):
#     '''
#     Pad non-word sequences.
#     :param sequences:
#     :return:
#     '''
#     lengths = [[len(seq)] for seq in sequences]
#     lengths_int = [len(seq) for seq in sequences]
#     # padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
#     # padded_seqs = tf.zeros([len(sequences), max(lengths)], dtype=tf.dtypes.float32)
#     padded_seqs = np.zeros([len(sequences), max(lengths_int)], dtype=np.dtype(np.float))
#     for i, seq in enumerate(sequences):
#         end = lengths_int[i]
#         padded_seqs[i, :end] = seq[:end]
#     return padded_seqs, lengths


# def padding_text(sequences):
#     '''
#     Pad plain information
#     :param sequences:
#     :return:
#     '''
#     lengths = [[len(seq)] for seq in sequences]
#     lengths_int = [len(seq) for seq in sequences]
#     new_sequences = []
#     for i, seq in enumerate(sequences):
#         length = lengths_int[i]
#         seq = seq + [['PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']] * (max(lengths_int) - length)
#         new_sequences.append(seq)
#     return new_sequences, lengths


# def padding_ent_index(sequences):
#     '''
#     Pad ent_index information.
#     :param sequences:
#     :return:
#     '''
#     lengths = [[len(seq)] for seq in sequences]
#     lengths_int = [len(seq) for seq in sequences]
#     new_sequences = []
#     for i, seq in enumerate(sequences):
#         length = lengths_int[i]
#         seq = seq + ['PAD'] * (max(lengths_int) - length)
#         new_sequences.append(seq)
#     return new_sequences, lengths


# def structure_transform(data):
#     '''
#     Transform data format to fit tensorflow.
#     :param data:
#     :return:
#     '''
#     data_info, data_info_processed = {}, {}
#     for key in data[0].keys():
#         data_info[key] = [d[key] for d in data]
#
#     context_arr, context_arr_lengths = padding(data_info['context_arr'], True)
#     response, response_lengths = padding(data_info['response'], False)
#     selector_index, _ = padding_index(data_info['selector_index'])
#     ptr_index, _ = padding(data_info['ptr_index'], False)
#     conv_arr, conv_arr_lengths = padding(data_info['conv_arr'], True)
#     sketch_response, sketech_response_lengths = padding(data_info['sketch_response'], False)
#     kb_arr, kb_arr_lengths = padding(data_info['kb_arr'], True)
#     context_arr_plain, context_arr_plain_lengths = padding_text(data_info['context_arr_plain'])
#     kb_arr_plain, kb_arr_plain_lengths = padding_text(data_info['kb_arr_plain'])
#     ent_index, ent_index_lengths = padding_ent_index(data_info['ent_index'])
#     ent_idx_cal, ent_idx_cal_lengths = padding_ent_index(data_info['ent_idx_cal'])
#     ent_idx_nav, ent_idx_nav_lengths = padding_ent_index(data_info['ent_idx_nav'])
#     ent_idx_wet, ent_idx_wet_lengths = padding_ent_index(data_info['ent_idx_wet'])
#
#
#     for key in data_info.keys():
#         try:
#             data_info_processed[key] = locals()[key]
#         except:
#             data_info_processed[key] = data_info[key]
#
#     # additional plain information
#     # print(np.array(context_arr_lengths).shape)
#     # print(np.array(response_lengths).shape)
#     # print(np.array(conv_arr_lengths).shape)
#     # print(np.array(kb_arr_lengths).shape)
#     # data_info_processed['context_arr_plain'] = np.array(data_info_processed['context_arr_plain'])
#     # print(data_info_processed['context_arr_plain'].shape)
#     data_info_processed['context_arr_lengths'] = np.array(context_arr_lengths)
#     data_info_processed['response_lengths'] = np.array(response_lengths)
#     data_info_processed['conv_arr_lengths'] = np.array(conv_arr_lengths)
#     data_info_processed['kb_arr_lengths'] = np.array(kb_arr_lengths)
#     data_info_processed['ent_index_lengths'] = np.array(ent_index_lengths)
#     data_info_processed['ent_idx_cal_lengths'] = np.array(ent_idx_cal_lengths)
#     data_info_processed['ent_idx_nav_lengths'] = np.array(ent_idx_nav_lengths)
#     data_info_processed['ent_idx_wet_lengths'] = np.array(ent_idx_wet_lengths)
#
#     return data_info_processed, max(sketech_response_lengths)[0]


# def build_dataset(data_info, batch_size):
#     '''
#     Build tensorflow dataset.
#     :param data_info:
#     :return:
#     '''
#     dataset = tf.data.Dataset.from_tensor_slices((data_info['context_arr'],
#                                                   data_info['response'],
#                                                   data_info['sketch_response'],
#                                                   data_info['conv_arr'],
#                                                   data_info['ptr_index'],
#                                                   data_info['selector_index'],
#                                                   data_info['kb_arr'],
#                                                   data_info['context_arr_plain'],
#                                                   data_info['response_plain'],
#                                                   data_info['kb_arr_plain'],
#                                                   data_info['context_arr_lengths'],
#                                                   data_info['response_lengths'],
#                                                   data_info['conv_arr_lengths'],
#                                                   data_info['kb_arr_lengths'],
#                                                   data_info['ent_index'],
#                                                   data_info['ent_index_lengths'],
#                                                   data_info['ent_idx_cal'],
#                                                   data_info['ent_idx_nav'],
#                                                   data_info['ent_idx_wet'],
#                                                   data_info['ent_idx_cal_lengths'],
#                                                   data_info['ent_idx_nav_lengths'],
#                                                   data_info['ent_idx_wet_lengths'],
#                                                   data_info['ID']
#                                                   )).shuffle(len(data_info['context_arr']))
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     return dataset


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/KVR/{}train.txt'.format(task)
    file_dev = 'data/KVR/{}dev.txt'.format(task)
    file_test = 'data/KVR/{}test.txt'.format(task)

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    # build lang (1.0, 2.0, 3.0)
    lang = build_lang(pair_train, True)

    # map word to ids (1.0, 2.0, 3.0)
    train_seq = text_to_sequence(pair_train, lang)
    dev_seq = text_to_sequence(pair_dev, lang)
    test_seq = text_to_sequence(pair_test, lang)

    # generate padded batches using tf.data.Dataset.from_generator (3.0)
    train = get_seq(train_seq, batch_size, drop_remainder=False)
    dev = get_seq(dev_seq, batch_size, drop_remainder=False)
    test = get_seq(test_seq, batch_size, drop_remainder=False)

    # debug dataset batch result
    context_arr, response, sketch_response, conv_arr, ptr_index, selector_index, kb_arr, context_arr_plain, response_plain,\
        kb_arr_plain, context_arr_lengths, response_lengths, conv_arr_lengths, kb_arr_lengths, ent_index, ent_index_lengths,\
        ent_idx_cal, ent_idx_nav, ent_idx_wet, ent_idx_cal_lengths, ent_idx_nav_lengths, ent_idx_wet_lengths, ID, deps, deps_type, cell_masks = next(iter(train))

    # structure transform, shuffle, batch, padding (previous version 2.0, batch padding, write by myself, deprecated)
    # train_samples = Dataset(train_seq, batch_size, shuffle=True)
    # train_samples_batches = train_samples.load_batches(drop_last=True)
    # dev_samples = Dataset(dev_seq, batch_size, shuffle=False)
    # dev_samples_batches = dev_samples.load_batches(drop_last=True)
    # test_samples = Dataset(test_seq, batch_size, shuffle=False)
    # test_samples_batches = test_samples.load_batches(drop_last=True)

    # # extract information from seqs (previous version 1.0, global padding, write by myself, deprecated)
    # train_info, train_max_resp_len = structure_transform(train_seq)
    # dev_info, dev_max_resp_len = structure_transform(dev_seq)
    # test_info, test_max_resp_len = structure_transform(test_seq)
    #
    # # build dataset (previous version 1.0, global padding, write by myself, deprecated)
    # train_samples = build_dataset(train_info, batch_size)
    # dev_samples = build_dataset(dev_info, batch_size)
    # test_samples = build_dataset(test_info, batch_size)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    # return train_samples, dev_samples, test_samples, [], lang, max_resp_len, len(pair_train), len(pair_dev), len(pair_test), train_max_resp_len, dev_max_resp_len, test_max_resp_len
    # return train_samples_batches, dev_samples_batches, test_samples_batches, [], lang, max_resp_len
    return train, dev, test, [], lang, max_resp_len, len(pair_train), len(pair_dev), len(pair_test)
