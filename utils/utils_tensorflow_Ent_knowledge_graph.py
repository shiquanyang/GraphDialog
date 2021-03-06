import json
import ast
import tensorflow as tf
from utils.utils_general import *
import numpy as np
from utils.tensorflow_dataset import *
from utils.utils_tensorflow_generator_kvr_knowledge_graph import *
from utils.utils_build_document_graph import *
import pdb


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], [], []
    node2id, neighbors_info = {}, {}
    node_cnt = 0
    max_resp_len = 0
    total_node_cnt, total_dep_cnt = 0, 0

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

                if '\t' in line:
                    # deal with dialogue history
                    nid, line = line.split(' ', 1)
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_arr_plain.append(u)

                    # Get gold entity
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = gold_ent
                    elif task_type == "schedule":
                        ent_idx_cal = gold_ent
                    elif task_type == "navigate":
                        ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get entity-head mapping
                    ent_head_mapping = {}
                    if task_type == "navigate":
                        for word_arr in kb_arr:
                            n = 0
                            for elm in word_arr:
                                if elm != 'PAD': n += 1
                            if n == 5:
                                head = word_arr[0]
                                entity = word_arr[0]
                            else:
                                head = word_arr[2]
                                entity = word_arr[0]
                            if entity not in ent_head_mapping:
                                ent_head_mapping[entity] = [head]
                            else:
                                ent_head_mapping[entity].append(head)
                    elif task_type == "weather":
                        for word_arr in kb_arr:
                            n = 0
                            for elm in word_arr:
                                if elm != 'PAD': n += 1
                            if n == 2: continue
                            elif n == 3:
                                head = word_arr[2]
                                entity = word_arr[0]
                            elif n == 4:
                                head = word_arr[3]
                                entity = word_arr[0]
                            else:
                                continue
                            if entity not in ent_head_mapping:
                                ent_head_mapping[entity] = [head]
                            else:
                                ent_head_mapping[entity].append(head)
                    elif task_type == "schedule":
                        if len(kb_arr) != 0:
                            for word_arr in kb_arr:
                                head = word_arr[2]
                                entity = word_arr[0]
                                if entity not in ent_head_mapping:
                                    ent_head_mapping[entity] = [head]
                                else:
                                    ent_head_mapping[entity].append(head)

                    # Get head-entity mapping
                    head_ent_mapping = {}
                    if ent_head_mapping:
                        for ent in ent_head_mapping.keys():
                            head_list = ent_head_mapping[ent]
                            for head in head_list:
                                if head not in head_ent_mapping:
                                    head_ent_mapping[head] = [ent]
                                else:
                                    if ent not in head_ent_mapping[head]:
                                        head_ent_mapping[head].append(ent)
                                    else:
                                        continue

                    # Get head pointer for words in response
                    r_list = r.split(' ')
                    head_lists = []
                    for word in r_list:
                        head_list = []
                        if word in ent_head_mapping:
                            for head in ent_head_mapping[word]:
                                if head not in head_list:
                                    head_list.append(head)
                        if head_list:
                            head_lists.append(head_list)
                    final_list = []
                    if head_lists:
                        final_list = head_lists[0]
                        for elm in head_lists:
                            final_list = list(set(final_list).intersection(set(elm)))

                    entity_list = []
                    for head in final_list:
                        if head in head_ent_mapping:
                            entities = head_ent_mapping[head]
                            for ent in entities:
                                if ent not in entity_list:
                                    entity_list.append(ent)
                    head_pointer = [1 if ((word_arr[0] in entity_list and set(final_list).intersection(set(word_arr)) != set([]) and '$u' not in word_arr and '$s' not in word_arr) or (word_arr[0] in r.split() and word_arr[0] in ent_index and ('$u' in word_arr or '$s' in word_arr))) else 0 for word_arr in context_arr] + [1]
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
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]
                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    # Document Graph
                    dep_info, dep_info_hat, max_len = dependency_parsing(conv_arr_plain)
                    dep_node_info, dep_relation_info, cell_mask, all_cnt, path_len_info = generate_subgraph(
                        dep_info_hat,
                        max_len,
                        False)
                    dep_node_info_reverse, dep_relation_info_reverse, cell_mask_reverse, all_cnt_reverse, path_len_info_reverse = generate_subgraph(
                        dep_info_hat,
                        max_len,
                        True)
                    deps = [dep_node_info, dep_node_info_reverse]
                    deps_type = [dep_relation_info, dep_relation_info_reverse]
                    masks = [cell_mask, cell_mask_reverse]
                    total_node_cnt = total_node_cnt + max_len
                    total_dep_cnt = total_dep_cnt + all_cnt + all_cnt_reverse

                    # Adjacent Matrix
                    adj = np.eye(len(context_arr) + 1)
                    for node in neighbors_info.keys():
                        neighbor = neighbors_info[node]
                        neighbor_list = neighbor.lstrip('[').rstrip(']').split(',')
                        neighbor = [ne.strip().strip('\'') for ne in neighbor_list]
                        node_id = (-1 * node2id[node]) + node_cnt - 1
                        for elm in neighbor:
                            elm_id = (-1 * node2id[elm]) + node_cnt - 1
                            adj[node_id, elm_id] = 1

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
                        'conv_arr_plain': list(conv_arr_plain),
                        'deps': list(deps),
                        'deps_type': list(deps_type),
                        'cell_masks': list(masks),
                        'kb_arr': list(kb_arr + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type,
                        'adj': list(adj),
                        'head_pointer': head_pointer}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_arr_plain.append(r)
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    # deal with knowledge graph
                    nid, node, neighbors = line.split('|')
                    r = node.lstrip('[').rstrip(']')
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
                    node2id[node] = node_cnt
                    node_cnt += 1
                    neighbors_info[node] = neighbors
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], []
                node2id, neighbors_info = {}, {}
                node_cnt = 0
                if (max_line and cnt_lin >= max_line):
                    break

    print('{} avg dependencies per node is: {}'.format(file_name, (total_dep_cnt/total_node_cnt)))
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
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
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
            lang.index_type(pair['deps_type'])
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


def preprocess_type(sequence, type2id):
    """
    Converts dependency types to ids.
    """
    deps_type = []
    for i, direction in enumerate(sequence):
        deps_type.append([])
        for ii, word_triple in enumerate(direction):
            deps_type[i].append([])
            for iii, word in enumerate(word_triple):
                temp = type2id[word] if word in type2id else UNK_token
                deps_type[i][ii].append(temp)
    deps_type = tf.convert_to_tensor(deps_type)
    return deps_type


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
        deps_type = preprocess_type(pair['deps_type'], lang.type2index)
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
            'ID':pair['ID'],
            'deps': pair['deps'],
            'deps_type': deps_type,
            'cell_masks': pair['cell_masks'],
            'adj': tf.convert_to_tensor(pair['adj']),
            'head_pointer': tf.convert_to_tensor(pair['head_pointer'])
        })
    return sequence_data


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/KVR/train_graph.txt'
    file_dev = 'data/KVR/dev_graph.txt'
    file_test = 'data/KVR/test_graph.txt'
    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    # build lang
    lang = build_lang(pair_train, True)

    # output edge-type dict
    # edge_type_cnt = lang.type2cnt
    # sorted_dict = sorted(edge_type_cnt.items(), key=lambda item: item[1], reverse=True)
    # with open('/Users/shiquan/PycharmProjects/GLMP_dev/GLMP/edge_type_stats_maxdeps_7.txt', 'w') as f:
    #     for key in sorted_dict:
    #         f.write("%s\t%s\n"%(key[0], key[1]))

    # map word to ids
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
        ent_idx_cal, ent_idx_nav, ent_idx_wet, ent_idx_cal_lengths, ent_idx_nav_lengths, ent_idx_wet_lengths, ID, deps, deps_type, cell_masks, adj, head_pointer = next(iter(train))

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len, len(pair_train), len(pair_dev), len(pair_test)
