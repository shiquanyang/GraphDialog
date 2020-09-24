import tensorflow as tf
from utils.config import *
from tensorflow_models.encoder import ContextRNN
from tensorflow_models.GraphGRU import GraphGRU
from tensorflow_models.KnowledgeGraph_with_gate import KnowledgeGraph
from tensorflow_models.decoder_with_gate import LocalMemoryDecoder
import random
import numpy as np
from tensorflow.python.framework import ops
import json
from utils.measures import wer, moses_multi_bleu
from utils.tensorflow_masked_cross_entropy import *
from utils.utils_general import *
import pdb

class GLMPGraph(tf.keras.Model):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, n_layers, graph_hidden_size, nheads, alpha, dropout, graph_dr, n_graph_layers):
        super(GLMPGraph, self).__init__()
        # self.name = 'GLMP'
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.graph_hidden_size = graph_hidden_size  # 8
        self.nheads = nheads  # 8
        self.alpha = alpha  # 0.2
        self.dropout = dropout
        self.graph_dr = graph_dr
        self.n_graph_layers = n_graph_layers
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = tf.keras.layers.Softmax(0)
        self.encoder = GraphGRU(lang.n_words, hidden_size, dropout, lang, (MAX_DEPENDENCIES_PER_NODE+1))
        self.extKnow = KnowledgeGraph(lang.n_words, hidden_size, n_layers, graph_hidden_size, nheads, alpha, graph_dr, n_graph_layers)
        self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang,
                                          hidden_size, self.decoder_hop, dropout)
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              extKnow=self.extKnow,
                                              decoder=self.decoder)
        if path:
            self.checkpoint.restore(path)  # path include: directory + prefix + id.

        self.encoder_optimizer = tf.keras.optimizers.Adam(lr)
        self.extKnow_optimizer = tf.keras.optimizers.Adam(lr)
        self.decoder_optimizer = tf.keras.optimizers.Adam(lr)
        # TODO: lr scheduler.

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_loss_g = tf.keras.metrics.Mean('train_loss_global', dtype=tf.float32)
        self.train_loss_v = tf.keras.metrics.Mean('train_loss_vocab', dtype=tf.float32)
        self.train_loss_l = tf.keras.metrics.Mean('train_loss_local', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

        self.reset()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, LE:{:.2f}, LG:{:.2f}, LP:{:.2f}'.format(
            print_loss_avg, print_loss_g, print_loss_v, print_loss_l)

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l = 0.0, 1.0, 0.0, 0.0, 0.0

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/GLMP-'+args["addName"]+name_data+str(self.task)+'HDD'+\
                    str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+\
                    'L'+layer_info+'lr'+str(self.lr)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        checkpoint_prefix = directory + '/ckpt'
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing,
                          get_decoded_words, training):
        # build unknown mask for memory if training mode
        if args['unk_mask'] and training:  # different: training flag need to be fed from outside explicitly.
            story_size = data[0].shape  # data[0]: context_arr.
            rand_mask = np.ones(story_size, dtype=np.float32)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]), dtype=np.float32)],
                                         1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            conv_rand_mask = np.ones(data[3].shape, dtype=np.float32)  # data[3]: conv_arr.
            for bi in range(story_size[0]):
                start, end = data[13][bi] - 1, data[13][bi] - 1 + data[12][bi]  # data[13]: kb_arr_lengths, data[12]: conv_arr_lengths.
                # start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]  # data[13]: kb_arr_lengths, data[12]: conv_arr_lengths.
                # conv_rand_mask[:end.numpy()[0]-start.numpy()[0], bi, :] = rand_mask[bi, start.numpy()[0]:end.numpy()[0], :]  # necessary to explictly move data to cuda ?
                conv_rand_mask[bi, :end-start, :] = rand_mask[bi, start:end, :]  # story_size dimension order is different from pytorch, so the slice index is different from pytorch one. necessary to explictly move data to cuda ?
            conv_story = data[3] * conv_rand_mask  # data[3]: conv_arr.
            story = data[0] * rand_mask  # data[0]: context_arr.
        else:
            story, conv_story = data[0], data[3]  # data[0]: context_arr, data[3]: conv_arr.

        # encode dialogue history and KB to vectors
        # TODO: need to check the shape and meaning of each tensor.
        dh_outputs, dh_hidden = self.encoder(conv_story, data[12], data[23], data[24], data[25], training=training)  # data[12]: conv_arr_lengths, data[23]: deps, data[24]: deps_type, data[25]: cell_masks.
        global_pointer, kb_readout, global_pointer_logits = self.extKnow.load_graph(story,
                                                                                    data[13],  # data[13]: kb_arr_lengths.
                                                                                    data[12],  # data[12]: conv_arr_lengths.
                                                                                    dh_hidden,
                                                                                    dh_outputs,
                                                                                    data[26],  # data[26]: adj.
                                                                                    training=training)
        encoded_hidden = tf.concat([dh_hidden, kb_readout], 1)
        # encoded_hidden = dh_hidden

        # get the words that can be copy from the memory
        batch_size = len(data[10])  # data[10]: context_arr_lengths.
        self.copy_list = []
        # pdb.set_trace()
        for elm in data[7]:  # data[7]: context_arr_plain.
            # elm_temp = [word_arr[0] for word_arr in elm]
            elm_temp = []
            for word_arr in elm:
                # elm_temp.append(word_arr[0])
                if word_arr[0].numpy().decode() != 'PAD':
                    elm_temp.append(word_arr[0].numpy().decode())
                else:
                #     self.copy_list.append(elm_temp)
                    break
            self.copy_list.append(elm_temp)
        # pdb.set_trace()
        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, outputs_gate_signal = self.decoder(self.extKnow,
                                                                                story.shape,
                                                                                data[10],  # data[10]: context_arr_lengths.
                                                                                self.copy_list,
                                                                                encoded_hidden,
                                                                                data[2],  # data[2]: sketch_response.
                                                                                max_target_length,
                                                                                batch_size,
                                                                                use_teacher_forcing,
                                                                                get_decoded_words,
                                                                                global_pointer,
                                                                                data[13],  # data[13]: kb_arr_lengths.
                                                                                data[12],  # data[12]: conv_arr_lengths.
                                                                                training=training)

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer, global_pointer_logits, outputs_gate_signal

    @tf.function
    def train_batch(self, data, clip, reset=0):
        # model training process
        # no need to zero gradients of optimizers in tensorflow
        # encode and decode
        with tf.GradientTape(persistent=True) as tape:
            # pdb.set_trace()
            use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
            max_target_length = max(data[11])  # data[11]: response_lengths.
            # max_target_length = train_max_len_global
            all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer, global_pointer_logits, all_decoder_outputs_gate_signal = self.encode_and_decode(data,
                                                                                                              max_target_length,
                                                                                                              use_teacher_forcing,
                                                                                                              False,
                                                                                                              True)
            # loss calculation and backpropagation
            # pdb.set_trace()
            loss_g = tf.cast(tf.compat.v1.losses.sigmoid_cross_entropy(data[5], tf.cast(global_pointer_logits, dtype=tf.double)), dtype=tf.float32)
            #loss_gs = tf.keras.backend.binary_crossentropy(tf.cast(data[5], dtype=tf.double), tf.cast(global_pointer, dtype=tf.double))
            #loss_g = tf.cast(tf.reduce_sum(loss_gs) / (loss_gs.shape[0]*loss_gs.shape[1]), dtype=tf.float32)
            # loss_g_mat = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(global_pointer, dtype=tf.double), data['selector_index'])  # data[5]: selector_index.
            # loss_g = tf.cast(tf.reduce_sum(loss_g_mat) / (loss_g_mat.shape[0] * loss_g_mat.shape[1]), dtype=tf.float32)
            # print("loss_g:", loss_g)
            loss_v = masked_cross_entropy(tf.transpose(all_decoder_outputs_vocab, [1, 0, 2]),  # need to transpose ?
                                          data[2],
                                          tf.cast(data[11], dtype=tf.int32))  # data[2]: skectch_response, data[11]: response_lengths.
            # print("loss_v:", loss_v)
            loss_l = masked_cross_entropy(tf.transpose(all_decoder_outputs_ptr, [1, 0, 2]),  # need to transpose ?
                                          data[4],
                                          tf.cast(data[11], dtype=tf.int32))  # data[4]: ptr_index, data[11]: response_lengths.
            loss_m = masked_cross_entropy(tf.transpose(all_decoder_outputs_gate_signal, [1, 0, 2]),
                                          data[27],
                                          tf.cast(data[11], dtype=tf.int32))  # data[27]: gate_label, data[11]: response_lengths.
            # print("loss_l:", loss_l)
            loss = loss_g + loss_v + 3 * loss_l + loss_m

        # compute gradients for encoder, decoder and external knowledge
        encoder_variables = self.encoder.trainable_variables
        extKnow_variables = self.extKnow.trainable_variables
        decoder_variables = self.decoder.trainable_variables
        encoder_gradients = tape.gradient(loss, encoder_variables)
        extKnow_gradients = tape.gradient(loss, extKnow_variables)
        decoder_gradients = tape.gradient(loss, decoder_variables)

        # clip gradients
        encoder_gradients, ec = tf.clip_by_global_norm(encoder_gradients, clip)
        extKnow_gradients, kc = tf.clip_by_global_norm(extKnow_gradients, clip)
        decoder_gradients, dc = tf.clip_by_global_norm(decoder_gradients, clip)
        # clipped_encoder_gradients = [elem if isinstance(elem, ops.IndexedSlices) else tf.clip_by_norm(elem, clip) for elem in encoder_gradients]
        # clipped_extKnow_gradients = [elem if isinstance(elem, ops.IndexedSlices) else tf.clip_by_norm(elem, clip) for elem in extKnow_gradients]
        # clipped_decoder_gradients = [elem if isinstance(elem, ops.IndexedSlices) else tf.clip_by_norm(elem, clip) for elem in decoder_gradients]

        # apply update
        self.encoder_optimizer.apply_gradients(
            zip(encoder_gradients, self.encoder.trainable_variables))
        self.extKnow_optimizer.apply_gradients(
            zip(extKnow_gradients, self.extKnow.trainable_variables))
        self.decoder_optimizer.apply_gradients(
            zip(decoder_gradients, self.decoder.trainable_variables))

        self.loss += loss.numpy()
        self.loss_g += loss_g.numpy()
        self.loss_v += loss_v.numpy()
        self.loss_l += loss_l.numpy()

        self.train_loss(loss.numpy())
        self.train_loss_g(loss_g.numpy())
        self.train_loss_v(loss_v.numpy())
        self.train_loss_l(loss_l.numpy())

    def evaluate(self, dev, dev_length, matric_best, early_stop=None):
        print('STARTING EVALUATION:')

        fd = open('test_result_{}.txt'.format(args['dataset']), 'a')

        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev.take(-1)), total=(dev_length))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        for j, data_dev in pbar:
            # Encode and Decode
            # pdb.set_trace()
            # max_target_length = max(data_dev['response_lengths'])  # data[11]: response_lengths.
            # max_target_length = dev_max_len_global
            _, _, decoded_fine, decoded_coarse, global_pointer, global_pointer_logits, _ = self.encode_and_decode(data_dev,
                                                                                        self.max_resp_len,
                                                                                        False,
                                                                                        True,
                                                                                        False)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                # pdb.set_trace()
                gold_sent = data_dev[8][bi][0].numpy().decode().lstrip().rstrip()  # data[8]: response_plain.
                ref.append(gold_sent)
                hyp.append(pred_sent)

                fd.write("predict response: " + pred_sent + "\n")
                fd.write("golden  response: " + gold_sent + "\n")
                fd.write("\n")

                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev[14][bi], pred_sent.split(),
                                                        global_entity_list, data_dev[9][bi])  # data[14]: ent_index, data[9]: kb_arr_plain.
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev[16][bi], pred_sent.split(),
                                                        global_entity_list, data_dev[9][bi])  # data[16]: ent_idx_cal, data[9]: kb_arr_plain.
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev[17][bi], pred_sent.split(),
                                                        global_entity_list, data_dev[9][bi])  # data[17]: ent_idx_nav, data[9]: kb_arr_plain.
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev[18][bi], pred_sent.split(),
                                                        global_entity_list, data_dev[9][bi])  # data[18]: ent_idx_wet, data[9]: kb_arr_plain.
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                else:
                    # compute Dialogue Accuracy Score
                    current_id = data_dev[22][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        fd.close()

        # pdb.set_trace()
        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred / float(F1_cal_count)))
            print("\tWET F1:\t{}".format(F1_wet_pred / float(F1_wet_count)))
            print("\tNAV F1:\t{}".format(F1_nav_pred / float(F1_nav_count)))
            print("BLEU SCORE:\t" + str(bleu_score))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            print("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        if (early_stop == 'BLEU'):
            # if (bleu_score >= matric_best):
            self.save_model('BLEU-' + str(bleu_score))
            print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            # if (F1_score >= matric_best):
            self.save_model('ENTF1-{:.4f}'.format(F1_score))
            print("MODEL SAVED")
            return F1_score
        else:
            # if (acc_score >= matric_best):
            self.save_model('ACC-{:.4f}'.format(acc_score))
            print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        # local_kb_word = [k[0] for k in kb_plain]
        # local_kb_word = [k[0] if k[0].decode() != '$' and k[0].decode() != 'PAD' for k in kb_plain]
        local_kb_word = []
        for k in kb_plain:
            if k[0].numpy().decode() != '$$$$' and k[0].numpy().decode() != 'PAD':
                local_kb_word.append(k[0].numpy().decode())
            else:
                break
        # gold_decode = [ent.decode() if ent.decode() != 'PAD' for ent in gold]
        gold_decode = []
        for ent in gold:
            if ent.numpy().decode() != 'PAD':
                gold_decode.append(ent.numpy().decode())
            else:
                break
        TP, FP, FN = 0, 0, 0
        if len(gold_decode) != 0:
            count = 1
            for g in gold_decode:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold_decode:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data['conv_arr_lengths'][batch_idx] - 1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w != 'PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr, ': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
