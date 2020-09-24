import spacy
from spacy.tokens import Doc
from utils.config import *
from spacy import displacy
from pathlib import Path
import copy
import numpy as np
import pdb


# customize tokenizer (white-space tokenizer)
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_parsing(sents):
    '''
    Parse the dependency relations at sentence level.
    :param sents: input sentences.
    :return: parsed dependencies info.
    '''
    dep_info, dep_info_hat = [], []
    local_2_global_dict = {}
    word_cnt = 0

    for sentence in sents:
        # gen global ids of words
        sentence_key = hash(sentence)
        if sentence_key not in local_2_global_dict:
            local_2_global_dict[sentence_key] = {}
        doc = nlp(sentence)
        # svg = displacy.render(doc, style='dep', jupyter=False)
        # file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
        # output_path = Path("/Users/shiquan/PycharmProjects/GLMP_dev/GLMP/img/" + file_name)
        # output_path.open("w", encoding="utf-8").write(svg)
        dependency = doc.to_json()
        dep_info.append(dependency)
        for index, word in enumerate(doc):
            local_2_global_dict[sentence_key][index] = word_cnt
            word_cnt += 1

    # map local id to global id in dep_info
    for sentence in dep_info:
        text = sentence['text'].strip()
        sentence_key = hash(text)
        tokens = []
        for word in sentence['tokens']:
            word_global_pos = local_2_global_dict[sentence_key][word['id']]
            head_global_pos = local_2_global_dict[sentence_key][word['head']]
            dep_relation = word['dep']
            tokens.append({
                'id':word_global_pos,
                'dep':dep_relation,
                'head':head_global_pos
            })
        dep_info_hat.append({
            'text':text,
            'tokens':tokens
        })

    # add order relation to dep_info
    text = ' '.join(sents)
    max_len = len(text.split(' '))
    # tokens = []
    # for pos in range(max_len-1):
    #     tokens.append({
    #         'id':pos,
    #         'dep':'pre',
    #         'head':pos+1
    #     })
    #     tokens.append({
    #         'id':pos+1,
    #         'dep':'next',
    #         'head':pos
    #     })
    # dep_info_hat.append({
    #     'text':'order_relation',
    #     'tokens':tokens
    # })
    # TODO: Reverse the arc from dependent to head

    return dep_info, dep_info_hat, max_len


def reverse_graph_info(graph_info):
    '''
    Reverse the arc from dependent to head.
    :param graph_info:
    :return:
    '''
    for content in graph_info:
        for entry in content['tokens']:
            id = entry['id']
            head = entry['head']
            temp = id
            id = head
            head = temp
            entry['id'] = id
            entry['head'] = head
    return graph_info


def generate_subgraph(graph_info, node_num, reverse=False):
    '''
    Generate subgraph for forward and backward calculation of Bi-GraphLSTM.
    :param graph_info: original graph information structure provided by spacy.
    :param node_num: max node number in current dialogue history.
    :param reverse: forward or backward.
    :return: forward subgraph or backward subgraph.
    '''
    dep_node_info, dep_relation_info, cell_mask, path_len_info = [], [], [], []
    all_dependency_cnt = 0
    graph_info_r = copy.deepcopy(graph_info)
    if args['reverse_graph']:
        graph_info_r = reverse_graph_info(graph_info_r)
    for index in range(node_num):
        dependencies = []
        path_len = []
        if not reverse:
            relations = ['NEXT']
        else:
            relations = ['PRE']
        masks = [1]
        # for backward word_id mapping
        if reverse:
            reversed_ids = list(reversed(range(node_num)))
	# parse all dependencies of nodes
        for content in graph_info_r:
            for token in content['tokens']:
                if not reverse:
                    # skip root node
                    if token['id'] == index and token['id'] > token['head']:
                        dependencies.append(str(token['head']))
                        relations.append(token['dep'])
                        masks.append(1)
                        path_len.append(abs(token['id'] - token['head']))
                    # add bi-directional graph
                    if token['id'] < index and token['head'] == index:
                        dependencies.append(str(token['id']))
                        relations.append(token['dep'] + '_reverse')
                        masks.append(1)
                        path_len.append(abs(token['id'] - token['head']))
                else:
                    # skip root node
                    if token['id'] == (node_num - index - 1) and token['id'] < token['head']:
                        # map head-id to forward style
                        dependencies.append(str(reversed_ids.index(token['head'])))
                        relations.append(token['dep'])
                        masks.append(1)
                        path_len.append(abs(token['id'] - token['head']))
                    # add bi-directional graph
                    if token['id'] > (node_num - index - 1) and token['head'] == (node_num - index - 1):
                        dependencies.append(str(reversed_ids.index(token['id'])))
                        relations.append(token['dep'] + '_reverse')
                        masks.append(1)
                        path_len.append(abs(token['id'] - token['head']))
	# TODO: prune node dependencies strategy
	# prune node dependencies
        all_dependency_cnt += len(dependencies)
        if len(dependencies) > MAX_DEPENDENCIES_PER_NODE:
            dependencies = dependencies[0: MAX_DEPENDENCIES_PER_NODE]
            relations = relations[0: (MAX_DEPENDENCIES_PER_NODE + 1)]
            masks = masks[0: (MAX_DEPENDENCIES_PER_NODE + 1)]
            path_len = path_len[0: MAX_DEPENDENCIES_PER_NODE]
        else:
            dependencies = dependencies + ['$'] * (MAX_DEPENDENCIES_PER_NODE - len(dependencies))
            relations = relations + ['$'] * (MAX_DEPENDENCIES_PER_NODE - len(relations) + 1)
            masks = masks + [0] * (MAX_DEPENDENCIES_PER_NODE - len(masks) + 1)
            path_len = path_len + [0] * (MAX_DEPENDENCIES_PER_NODE - len(path_len))
        dep_node_info.append(dependencies)
        dep_relation_info.append(relations)
        cell_mask.append(masks)
        path_len_info.append(path_len)
    return dep_node_info, dep_relation_info, cell_mask, all_dependency_cnt, path_len_info


if __name__ == "__main__":
    conv_arr = ['what gas_stations are here ?', 'There is a chevron']
    dep_info, dep_info_hat, max_len = dependency_parsing(conv_arr)
    dep_node_info, dep_relation_info, cell_mask, all_cnt = generate_subgraph(
        dep_info_hat,
        max_len,
        False)
    dep_node_info_reverse, dep_relation_info_reverse, cell_mask_reverse, all_cnt = generate_subgraph(
        dep_info_hat,
        max_len,
        True)
    print(dep_info)
    print(dep_info_hat)
    print(dep_node_info)
    print(dep_relation_info)
    print(cell_mask)
    print(dep_node_info_reverse)
    print(dep_relation_info_reverse)
    print(cell_mask_reverse)

