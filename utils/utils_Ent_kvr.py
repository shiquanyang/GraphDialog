import json
import ast
import numpy as np
from utils.utils_general import *


def read_langs(file_name, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], [], []
    max_resp_len = 0
    node2id, neighbors_info = {}, {}
    node_cnt = 0
    context_debug = []
    
    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)
    
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#","")
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
                    temp_list = []
                    temp_list.append(u)
                    context_debug = context_debug + temp_list
                    
                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather": ent_idx_wet = gold_ent
                    elif task_type == "schedule": ent_idx_cal = gold_ent
                    elif task_type == "navigate": ent_idx_nav = gold_ent
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
                    # If it's the head's entity, Then label 1, Else label 0.
                    # head_pointer = [1 if word_arr[0] in entity_list and set(final_list).intersection(set(word_arr)) != set([]) and '$u' not in word_arr and '$s' not in word_arr else 0 for word_arr in context_arr] + [0]
                    head_pointer = [1 if ((word_arr[0] in entity_list and set(final_list).intersection(set(word_arr)) != set([]) and '$u' not in word_arr and '$s' not in word_arr) or (word_arr[0] in r.split() and word_arr[0] in ent_index and ('$u' in word_arr or '$s' in word_arr))) else 0 for word_arr in context_arr] + [1]

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index) 
                        else: 
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in context_arr] + [1]
                    
                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    gate_label = []
                    for key in r.split():
                        is_dialogue_history = 0
                        if key not in ent_index:
                            gate_label.append(2)
                        else:
                            for loc, val in enumerate(conv_arr):
                                if val[0] == key:
                                    gate_label.append(1)
                                    is_dialogue_history = 1
                                    break
                            if is_dialogue_history == 0:
                                gate_label.append(0)

                    # generate adjacent matrix
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
                        'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]), # $$$$ is NULL token
                        'response':r,
                        'sketch_response':sketch_response,
                        'ptr_index':ptr_index+[len(context_arr)],
                        'selector_index':selector_index,
                        'ent_index':ent_index,
                        'ent_idx_cal':list(set(ent_idx_cal)),
                        'ent_idx_nav':list(set(ent_idx_nav)),
                        'ent_idx_wet':list(set(ent_idx_wet)),
                        'conv_arr':list(conv_arr),
                        'kb_arr':list(kb_arr), 
                        'id':int(sample_counter),
                        'ID':int(cnt_lin),
                        'domain':task_type,
                        'adj': list(adj),
                        'gate_label': gate_label+[2],
                        'head_pointer': head_pointer,
                        'context_debug': list(context_debug)}
                    data.append(data_detail)
                    
                    gen_r = generate_memory(r, "$s", str(nid)) 
                    context_arr += gen_r
                    conv_arr += gen_r
                    temp_list_r = []
                    temp_list_r.append(r)
                    context_debug = context_debug + temp_list_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    nid, node, neighbors = line.split('|')
                    r = node.lstrip('[').rstrip(']')
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
                    node2id[node] = node_cnt
                    node_cnt += 1
                    neighbors_info[node] = neighbors
                    r = "#" + r
                    kb_temp = []
                    kb_temp.append(r)
                    context_debug = kb_temp + context_debug
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                context_debug = []
                node2id, neighbors_info = {}, {}
                node_cnt = 0
                if(max_line and cnt_lin >= max_line):
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
                        if key!='poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                sketch_response.append('@'+ent_type)        
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker=="$u" or speaker=="$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/KVR/{}train_transformed_full_features.txt'.format(task)
    file_dev = 'data/KVR/{}dev_transformed_full_features.txt'.format(task)
    # file_test = 'data/KVR/{}test_transformed_full_features.txt'.format(task)
    file_test = 'data/KVR/test_mock.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev   = get_seq(pair_dev, lang, batch_size, False)
    test  = get_seq(pair_test, lang, batch_size, False)
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d