import numpy as np

dataset = "test"
input_path = '../data/MULTIWOZ2.1/{}_w_kb_w_gold.txt'.format(dataset)
output_path = '../data/MULTIWOZ2.1/{}_edge_paris_full_features.txt'.format(dataset)
fd = open(output_path, 'w')

nsample = 0
kb_cnt = 0
is_output = 0
kb_info = []

with open(input_path) as f:
    for line in f:
        line = line.strip()
        if line:
            # if nsample == 12:
            #     print("stop")
            if line.startswith("#"):
                fd.write('#' + str(nsample) + '\n')
                line = line.replace('#', '')
                task_type = line
                continue
            nid, line = line.split(' ', 1)
            if nid == '0':
                kb_cnt += 1
                kb_info.append(line)
            else:
                # iterate generate node pairs
                if kb_cnt == 0:
                    continue
                if is_output == 1:
                    continue
                if task_type == 'navigate':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) == 5:
                            ent = triple_list[-1]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if len(elm_list) == 3 and elm_list[0] == ent:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        elif len(triple_list) == 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if len(elm_list) == 5 and elm_list[-1] == head:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        else:
                            continue
                    is_output = 1
                elif task_type == 'schedule':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        head = triple_list[0]
                        for elm in kb_info:
                            elm_list = elm.split(' ')
                            if elm_list[0] == head and tripe != elm:
                                fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                            else:
                                continue
                    is_output = 1
                elif task_type == 'weather':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) >= 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == head and tripe != elm:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                    is_output = 1
                elif task_type in ('restaurant', 'hotel', 'attraction', 'train', 'hospital'):
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) == 1:
                            ent = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == ent and len(elm_list) == 3:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        elif len(triple_list) == 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == head and len(elm_list) == 1:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                    is_output = 1
                else:
                    continue
        else:
            if kb_cnt == 0:
                fd.write('[]' + '\n')
            if is_output == 0:
                if task_type == 'navigate':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) == 5:
                            ent = triple_list[-1]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if len(elm_list) == 3 and elm_list[0] == ent:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        elif len(triple_list) == 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if len(elm_list) == 5 and elm_list[-1] == head:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        else:
                            continue
                    is_output = 1
                elif task_type == 'schedule':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        head = triple_list[0]
                        for elm in kb_info:
                            elm_list = elm.split(' ')
                            if elm_list[0] == head and tripe != elm:
                                fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                            else:
                                continue
                    is_output = 1
                elif task_type == 'weather':
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) >= 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == head and tripe != elm:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                    is_output = 1
                elif task_type in ('restaurant', 'hotel', 'attraction', 'train', 'hospital'):
                    for tripe in kb_info:
                        triple_list = tripe.split(' ')
                        if len(triple_list) == 1:
                            ent = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == ent and len(elm_list) == 3:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                        elif len(triple_list) == 3:
                            head = triple_list[0]
                            for elm in kb_info:
                                elm_list = elm.split(' ')
                                if elm_list[0] == head and len(elm_list) == 1:
                                    fd.write('[{}],[{}]'.format(tripe, elm) + '\n')
                                else:
                                    continue
                    is_output = 1
                else:
                    continue
            fd.write('\n')
            kb_info = []
            nsample += 1
            kb_cnt = 0
            is_output = 0

print('success.')