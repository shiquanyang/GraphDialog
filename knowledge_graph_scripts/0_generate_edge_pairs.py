import numpy as np

fd = open('../data/KVR/test_edge_paris.txt', 'w')

n_sample = 0
kb_cnt = 0

with open('../data/KVR/test.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            if '#' in line:
                fd.write('#' + str(n_sample) + '\n')
                line = line.replace('#', '')
                task_type = line
                continue
            nid, line = line.split(' ', 1)
            if nid == '0':
                kb_cnt += 1
                line_list = line.split(' ')
                if task_type == 'navigate':
                    if len(line_list) == 5:
                        continue
                    elif len(line_list) == 3:
                        fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                        fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                    else:
                        continue
                elif task_type == 'schedule':
                    fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                    fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                elif task_type == 'weather':
                    if len(line_list) == 3:
                        fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                        fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                    elif len(line_list) == 4:
                        fd.write('[{}],[{} {} {}]'.format(line_list[0], line_list[1], line_list[2], line_list[3]) + '\n')
                        fd.write('[{} {} {}],[{}]'.format(line_list[1], line_list[2], line_list[3], line_list[0]) + '\n')
                    else:
                        continue
        else:
            if kb_cnt == 0:
                fd.write('[]' + '\n')
            kb_cnt = 0
            fd.write('\n')
            n_sample += 1

print('success.')