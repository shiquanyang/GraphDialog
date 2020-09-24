import numpy as np
import matplotlib.pyplot as plt

epochs_1 = []
total_losses_1 = []
global_pointer_losses_1 = []
generation_losses_1 = []
local_pointer_losses_1 = []
acc_scores_1 = []
f1_scores_1 = []
cal_f1_scores_1 = []
wet_f1_scores_1 = []
nav_f1_scores_1 = []
bleus_1 = []

epochs_2 = []
total_losses_2 = []
global_pointer_losses_2 = []
generation_losses_2 = []
local_pointer_losses_2 = []
acc_scores_2 = []
f1_scores_2 = []
cal_f1_scores_2 = []
wet_f1_scores_2 = []
nav_f1_scores_2 = []
bleus_2 = []

with open('../tmp/result3.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('Epoch:'):
            epoch = line.strip().split(':')[1]
            epochs_1.append(int(epoch))
        elif line.startswith('L:'):
            losses = line.strip().split(',')
            total_loss = losses[0].strip().split(':')[1]
            global_pointer_loss = losses[1].strip().split(':')[1]
            generation_loss = losses[2].strip().split(':')[1]
            local_pointer_loss = losses[3].strip().split(':')[1]
            global_pointer_losses_1.append(float(global_pointer_loss))
            generation_losses_1.append(float(generation_loss))
            local_pointer_losses_1.append(float(local_pointer_loss))
            total_losses_1.append(float(total_loss))
        elif line.startswith('ACC SCORE:'):
            scores = line.strip().split(':')
            acc = scores[1].strip()
            acc_scores_1.append(float(acc))
        elif line.startswith('F1 SCORE:'):
            scores = line.strip().split(':')
            f1 = scores[1].strip()
            f1_scores_1.append(float(f1))
        elif line.startswith('CAL F1:'):
            scores = line.strip().split(':')
            cal_f1 = scores[1].strip()
            cal_f1_scores_1.append(float(cal_f1))
        elif line.startswith('WET F1:'):
            scores = line.strip().split(':')
            wet_f1 = scores[1].strip()
            wet_f1_scores_1.append(float(wet_f1))
        elif line.startswith('NAV F1:'):
            scores = line.strip().split(':')
            nav_f1 = scores[1].strip()
            nav_f1_scores_1.append(float(nav_f1))
        elif line.startswith('BLEU SCORE:'):
            scores = line.strip().split(':')
            bleu = scores[1].strip()
            bleus_1.append(float(bleu))
        else:
            continue

with open('../tmp/result4.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('Epoch:'):
            epoch = line.strip().split(':')[1]
            epochs_2.append(int(epoch))
        elif line.startswith('L:'):
            losses = line.strip().split(',')
            total_loss = losses[0].strip().split(':')[1]
            global_pointer_loss = losses[1].strip().split(':')[1]
            generation_loss = losses[2].strip().split(':')[1]
            local_pointer_loss = losses[3].strip().split(':')[1]
            global_pointer_losses_2.append(float(global_pointer_loss))
            generation_losses_2.append(float(generation_loss))
            local_pointer_losses_2.append(float(local_pointer_loss))
            total_losses_2.append(float(total_loss))
        elif line.startswith('ACC SCORE:'):
            scores = line.strip().split(':')
            acc = scores[1].strip()
            acc_scores_2.append(float(acc))
        elif line.startswith('F1 SCORE:'):
            scores = line.strip().split(':')
            f1 = scores[1].strip()
            f1_scores_2.append(float(f1))
        elif line.startswith('CAL F1:'):
            scores = line.strip().split(':')
            cal_f1 = scores[1].strip()
            cal_f1_scores_2.append(float(cal_f1))
        elif line.startswith('WET F1:'):
            scores = line.strip().split(':')
            wet_f1 = scores[1].strip()
            wet_f1_scores_2.append(float(wet_f1))
        elif line.startswith('NAV F1:'):
            scores = line.strip().split(':')
            nav_f1 = scores[1].strip()
            nav_f1_scores_2.append(float(nav_f1))
        elif line.startswith('BLEU SCORE:'):
            scores = line.strip().split(':')
            bleu = scores[1].strip()
            bleus_2.append(float(bleu))
        else:
            continue

assert len(epochs_1) == len(total_losses_1)
assert len(epochs_1) == len(global_pointer_losses_1)
assert len(epochs_1) == len(generation_losses_1)
assert len(epochs_1) == len(local_pointer_losses_1)
assert len(epochs_1) == len(acc_scores_1)
assert len(epochs_1) == len(f1_scores_1)
assert len(epochs_1) == len(cal_f1_scores_1)
assert len(epochs_1) == len(wet_f1_scores_1)
assert len(epochs_1) == len(nav_f1_scores_1)
assert len(epochs_1) == len(bleus_1)

assert len(epochs_2) == len(total_losses_2)
assert len(epochs_2) == len(global_pointer_losses_2)
assert len(epochs_2) == len(generation_losses_2)
assert len(epochs_2) == len(local_pointer_losses_2)
assert len(epochs_2) == len(acc_scores_2)
assert len(epochs_2) == len(f1_scores_2)
assert len(epochs_2) == len(cal_f1_scores_2)
assert len(epochs_2) == len(wet_f1_scores_2)
assert len(epochs_2) == len(nav_f1_scores_2)
assert len(epochs_2) == len(bleus_2)

plt.figure('F1 Training Results')
plt.subplot(221)
plt.xlabel('Epoches')
plt.ylabel('f1_scores')
plt.plot(epochs_1, f1_scores_1, 'r-o')
plt.plot(epochs_2, f1_scores_2, 'b-^')
plt.subplot(222)
plt.xlabel('Epoches')
plt.ylabel('cal_f1_scores')
plt.plot(epochs_1, cal_f1_scores_1, 'r-o')
plt.plot(epochs_2, cal_f1_scores_2, 'b-^')
plt.subplot(223)
plt.xlabel('Epoches')
plt.ylabel('wet_f1_scores')
plt.plot(epochs_1, wet_f1_scores_1, 'r-o')
plt.plot(epochs_2, wet_f1_scores_2, 'b-^')
plt.subplot(224)
plt.xlabel('Epoches')
plt.ylabel('nav_f1_scores')
plt.plot(epochs_1, nav_f1_scores_1, 'r-o')
plt.plot(epochs_2, nav_f1_scores_2, 'b-^')

plt.figure('BLEU Training Results')
plt.xlabel('Epoches')
plt.ylabel('BLEU Score')
plt.plot(epochs_1, bleus_1, 'r-o')
plt.plot(epochs_2, bleus_2, 'b-^')

plt.figure('Losses Training Results')
plt.subplot(221)
plt.xlabel('Epoches')
plt.ylabel('total_losses')
plt.plot(epochs_1, total_losses_1, 'r-o')
plt.plot(epochs_2, total_losses_2, 'b-^')
plt.subplot(222)
plt.xlabel('Epoches')
plt.ylabel('global_pointer_losses')
plt.plot(epochs_1, global_pointer_losses_1, 'r-o')
plt.plot(epochs_2, global_pointer_losses_2, 'b-^')
plt.subplot(223)
plt.xlabel('Epoches')
plt.ylabel('local_pointer_losses')
plt.plot(epochs_1, local_pointer_losses_1, 'r-o')
plt.plot(epochs_2, local_pointer_losses_2, 'b-^')
plt.subplot(224)
plt.xlabel('Epoches')
plt.ylabel('generation_losses')
plt.plot(epochs_1, generation_losses_1, 'r-o')
plt.plot(epochs_2, generation_losses_2, 'b-^')
plt.show()