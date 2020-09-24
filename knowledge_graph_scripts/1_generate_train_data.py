import numpy as np
import json

edge_dict = {}

with open('../data/KVR/test_edge_paris.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            if '#' in line:
                line = line.replace('#', '')
                n_sample = line
                edge_dict[n_sample] = {}
                continue
            if line != '[]':
                head, tail = line.split(',')
                if head not in edge_dict[n_sample]:
                    edge_dict[n_sample][head] = []
                    edge_dict[n_sample][head].append(tail)
                else:
                    edge_dict[n_sample][head].append(tail)
            else:
                continue
        else:
            continue
print('success.')

fd = open('../data/KVR/test_transformed.txt', 'w')

node_list = []
n_sample = 0

with open('../data/KVR/test.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            if '#' in line:
                fd.write(line + '\n')
                line = line.replace('#', '')
                task_type = line
                continue
            nid, line = line.split(' ', 1)
            if nid == '0':  # KB
                line_list = line.split(' ')
                if task_type == 'navigate':
                    if len(line_list) == 5:
                        if list([line_list[4]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{}]'.format(line_list[4])]
                            fd.write('0|[{}]|{}'.format(line_list[4], neighbors) + '\n')
                            node_list.append([line_list[4]])
                        else:
                            continue
                    elif len(line_list) == 3:
                        if list([line_list[1], line_list[2]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{} {}]'.format(line_list[1], line_list[2])]
                            fd.write('0|[{} {}]|{}'.format(line_list[1], line_list[2], neighbors) + '\n')
                            node_list.append([line_list[1], line_list[2]])
                    else:
                        continue
                elif task_type == 'schedule':
                    if list([line_list[0]]) not in node_list:
                        neighbors = edge_dict[str(n_sample)]['[{}]'.format(line_list[0])]
                        fd.write('0|[{}]|{}'.format(line_list[0], neighbors) + '\n')
                        node_list.append([line_list[0]])
                    if list([line_list[1], line_list[2]]) not in node_list:
                        neighbors = edge_dict[str(n_sample)]['[{} {}]'.format(line_list[1], line_list[2])]
                        fd.write('0|[{} {}]|{}'.format(line_list[1], line_list[2], neighbors) + '\n')
                        node_list.append([line_list[1], line_list[2]])
                elif task_type == 'weather':
                    if len(line_list) == 3:
                        if list([line_list[0]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{}]'.format(line_list[0])]
                            fd.write('0|[{}]|{}'.format(line_list[0], neighbors) + '\n')
                            node_list.append([line_list[0]])
                        if list([line_list[1], line_list[2]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{} {}]'.format(line_list[1], line_list[2])]
                            fd.write('0|[{} {}]|{}'.format(line_list[1], line_list[2], neighbors) + '\n')
                            node_list.append([line_list[1], line_list[2]])
                    elif len(line_list) == 4:
                        if list([line_list[0]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{}]'.format(line_list[0])]
                            fd.write('0|[{}]|{}'.format(line_list[0], neighbors) + '\n')
                            node_list.append([line_list[0]])
                        if list([line_list[1], line_list[2], line_list[3]]) not in node_list:
                            neighbors = edge_dict[str(n_sample)]['[{} {} {}]'.format(line_list[1], line_list[2], line_list[3])]
                            fd.write('0|[{} {} {}]|{}'.format(line_list[1], line_list[2], line_list[3], neighbors) + '\n')
                            node_list.append([line_list[1], line_list[2], line_list[3]])
                    else:
                        continue
                else:
                    continue
            else:
                fd.write('{} {}'.format(nid, line) + '\n')
        else:
            fd.write('\n')
            node_list = []
            n_sample += 1
    print('success.')