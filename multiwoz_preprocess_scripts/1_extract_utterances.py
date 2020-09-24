import numpy as np
import json

valList = []
testList = []
with open('../data/MULTIWOZ2.1/valListFile.json') as f:
    for line in f:
        valList.append(line.replace('\n', ''))

with open('../data/MULTIWOZ2.1/testListFile.json') as f:
    for line in f:
        testList.append(line.replace('\n', ''))

fout = open('../data/MULTIWOZ2.1/train.txt', 'w')
fout_val = open('../data/MULTIWOZ2.1/valid.txt', 'w')
fout_test = open('../data/MULTIWOZ2.1/test.txt', 'w')

with open('../data/MULTIWOZ2.1/data.json') as f:
    data = json.load(f)
    for key in data.keys():
        if 'SNG' in key or 'WOZ' in key:  # single-domain data
            goal = data[key]['goal']
            log = data[key]['log']
            for domain in goal.keys():
                # drop out police and taxi data because of small data size.
                if goal[domain] and domain in ('taxi', 'police', 'hospital', 'hotel', 'attraction', 'restaurant', 'train'):
                    type = domain
            if type in ('taxi', 'police', 'hospital'):
                continue
            if key in valList:
                fout_val.write('#' + type + '#' + '\n')
            elif key in testList:
                fout_test.write('#' + type + '#' + '\n')
            else:
                fout.write('#' + type + '#' + '\n')
            cnt = 0
            for id, elm in enumerate(log):
                cnt += 1
                text = elm['text']
                text = text.replace("\n", " ")
                word_new = []
                # process punctuations
                for word in text.split(' '):
                    # to lowe case
                    word = word.lower()
                    if '\'' in word:
                        word = word.replace('\'', ' ')
                    if word.endswith(','):
                        word = word.replace(',', '')
                        word_new.append(word)
                        # word_new.append(',')
                    elif word.endswith('.'):
                        word = word.replace('.', '')
                        word_new.append(word)
                        # word_new.append('.')
                    elif word.endswith('?'):
                        word = word.replace('?', '')
                        word_new.append(word)
                        word_new.append('?')
                    elif word.endswith('!'):
                        word = word.replace('!', '')
                        word_new.append(word)
                        word_new.append('!')
                    else:
                        word_new.append(word)
                text = ' '.join(word_new)
                text = text.replace("\t", " ")
                text = ' '.join(text.split())
                if id % 2 == 0:
                    turn = int(np.floor(cnt / 2) + 1)
                    if key in valList:
                        fout_val.write(str(turn) + ' ' + text + '\t')
                    elif key in testList:
                        fout_test.write(str(turn) + ' ' + text + '\t')
                    else:
                        fout.write(str(turn) + ' ' + text + '\t')
                else:
                    if key in valList:
                        fout_val.write(text)
                    elif key in testList:
                        fout_test.write(text)
                    else:
                        fout.write(text)
                if (cnt % 2 == 0):
                    if key in valList:
                        fout_val.write('\n')
                    elif key in testList:
                        fout_test.write('\n')
                    else:
                        fout.write('\n')
            if key in valList:
                fout_val.write('\n')
            elif key in testList:
                fout_test.write('\n')
            else:
                fout.write('\n')
    print('success.')
