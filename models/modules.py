import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
from models.GraphAttentionLayer import GraphAttentionLayer
from torch.autograd import Variable
import numpy as np
import pdb


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2*hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # print("input_seqs in size: ", input_seqs.size())
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long()) 
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2) 
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        # print("input_seqs out size: ", input_seqs.size())
        # print("embedded size: ", embedded.size())
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0,1), hidden


class ExternalKnowledge4Head(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, nhid, nheads, alpha, dropout, graph_layer_num):
        super(ExternalKnowledge4Head, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.nhid = nhid
        self.nheads = nheads
        self.alpha = alpha
        self.graph_layer_num = graph_layer_num
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            # t = torch.randn(vocab, embedding_dim) * 0.1
            # t[PAD_token, :] = torch.zeros(1, embedding_dim)
            # C.weight.data = t
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def update_pad_token_adj(self, adj, kb_len, conv_len):
        batch_size = adj.shape[0]
        max_len = adj.shape[1]
        adj_array = adj.numpy()
        for i in range(batch_size):
            kb_len_i = kb_len[i] - 1
            conv_len_i = conv_len[i]
            context_len_i = kb_len_i + conv_len_i + 1
            for k in range(context_len_i, max_len):
                adj_array[i, k, k] = 1
        ret_adj = torch.Tensor(adj_array)
        return ret_adj

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, adj):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        adj = self.update_pad_token_adj(adj, kb_len, conv_len)
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)

        return self.sigmoid(prob_logit), u[-1]

class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, nhid, nheads, alpha, dropout, graph_layer_num):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.nhid = nhid
        self.nheads = nheads
        self.alpha = alpha
        self.graph_layer_num = graph_layer_num
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            # t = torch.randn(vocab, embedding_dim) * 0.1
            # t[PAD_token, :] = torch.zeros(1, embedding_dim)
            # C.weight.data = t
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.graph_layers_list = []
        for i in range(self.graph_layer_num):
            graph_layers = []
            for _ in range(self.max_hops+1):
                graph_layer = GraphAttentionLayer(embedding_dim, nhid, dropout, alpha, concat=False)
                graph_layers.append(graph_layer)
            self.graph_layers_list.append(graph_layers)

        self.W1 = nn.Linear(2 * embedding_dim, 4 * embedding_dim)
        self.W2 = nn.Linear(4 * embedding_dim, 1)
        # self.W2 = nn.Linear(4 * embedding_dim, 4 * embedding_dim)
        # self.W3 = nn.Linear(4 * embedding_dim, 2 * embedding_dim)
        # self.W4 = nn.Linear(2 * embedding_dim, 1)
        # self.W5 = nn.Linear(4 * embedding_dim, 2 * embedding_dim)
        # self.W6 = nn.Linear(2 * embedding_dim, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi]+conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def update_pad_token_adj(self, adj, kb_len, conv_len):
        batch_size = adj.shape[0]
        max_len = adj.shape[1]
        adj_array = adj.numpy()
        for i in range(batch_size):
            kb_len_i = kb_len[i] - 1
            conv_len_i = conv_len[i]
            context_len_i = kb_len_i + conv_len_i + 1
            for k in range(context_len_i, max_len):
                adj_array[i, k, k] = 1
        ret_adj = torch.Tensor(adj_array)
        return ret_adj

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs, adj):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        adj = self.update_pad_token_adj(adj, kb_len, conv_len)
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            # for layer in range(self.graph_layer_num):
            #     graph_layer = self.graph_layers_list[layer][hop]
            #     embed_A = graph_layer(embed_A, adj)
                # embed_A_t = graph_layer(embed_A, adj)
                # embed_A = embed_A + embed_A_t
            embed_A = self.dropout_layer(embed_A)
            
            if(len(list(u[-1].size()))==1): 
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A*u_temp, 2)
            prob_   = self.softmax(prob_logit)
            
            embed_C = self.C[hop+1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)
            # for layer in range(self.graph_layer_num):
            #     graph_layer = self.graph_layers_list[layer][hop+1]
            #     embed_C = graph_layer(embed_C, adj)
                # embed_C_t = graph_layer(embed_C, adj)
                # embed_C = embed_C + embed_C_t

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k  = torch.sum(embed_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        # additive attention: embed_A, u_temp
        # global_pointer = self.tanh(embed_A + u_temp) @ self.a1
        # head_pointer = self.tanh(embed_A + u_temp) @ self.a2
        # global_pointer = self.W4(self.tanh(self.W3(self.tanh(self.W2(self.tanh(self.W1(torch.cat([embed_A, u_temp], 2))))))))
        head_pointer = self.W1(torch.cat([embed_A, u_temp], 2))
        head_pointer = self.dropout_layer(head_pointer)
        head_pointer = self.W2(head_pointer)

        return self.sigmoid(prob_logit), u[-1], self.sigmoid(head_pointer.squeeze())
        # return self.sigmoid(prob_logit), u[-1]

    def forward(self, query_vector, global_pointer, gate_signal, kb_len, conv_len, head_pointer):
        u = [query_vector]

        # max_len = global_pointer.shape[1]  # global_pointer: batch_size * max_len.
        # batch_size = global_pointer.shape[0]
        # gate_signal_new = torch.zeros([batch_size, max_len])
        # for bi in range(gate_signal.shape[0]):
        #     kb_len_t = kb_len[bi]
        #     conv_len_t = conv_len[bi]
        #     kb_signal_t = gate_signal[bi, 0].unsqueeze(0).expand([kb_len_t])
        #     conv_signal_t = gate_signal[bi, 1].unsqueeze(0).expand([conv_len_t])
        #     null_signal_t = gate_signal[bi, 2].unsqueeze(0)
        #     pad_signal_t = torch.zeros([max_len - kb_len_t - conv_len_t - 1])
        #     gate_signal_t = torch.cat([kb_signal_t, conv_signal_t, null_signal_t, pad_signal_t], 0)
        #     gate_signal_new[bi, :] = gate_signal_t

        # global_pointer = global_pointer + gate_signal_new

        # global_pointer = self.a3 * global_pointer + self.a4 * head_pointer
        # global_pointer = global_pointer + head_pointer

        for hop in range(self.max_hops):
            m_A = self.m_story[hop] 
            if not args["ablationG"]:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if(len(list(u[-1].size()))==1):
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A*u_temp, 2)
            prob_soft   = self.softmax(prob_logits)
            m_C = self.m_story[hop+1] 
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        self.C = shared_emb 
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.W1 = nn.Linear(embedding_dim, 2*embedding_dim)
        self.W2 = nn.Linear(2*embedding_dim, 3)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length, batch_size, use_teacher_forcing, get_decoded_words, global_pointer, kb_len, conv_len, head_pointer, story):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        all_decoder_outputs_gate_signal = _cuda(torch.zeros(max_target_length, batch_size, 3))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)

        decoded_words = []
        self.from_whichs = []
        # Start to generate word-by-word
        for t in range(max_target_length):
            temp = self.C(decoder_input)
            embed_q = self.dropout_layer(self.C(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0] 
            # pdb.set_trace()
            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)

            gate_signal = self.relu(self.W1(query_vector))
            gate_signal = self.dropout_layer(gate_signal)
            gate_signal = self.W2(gate_signal)
            gate_signal = self.softmax(gate_signal)
            all_decoder_outputs_gate_signal[t] = gate_signal

            # query the external konwledge using the hidden state of sketch RNN
            prob_soft, prob_logits = extKnow(query_vector, global_pointer, gate_signal, kb_len, conv_len, head_pointer)
            all_decoder_outputs_ptr[t] = prob_logits
            _, toppi = prob_logits.data.topk(1)

            if use_teacher_forcing:
                decoder_input = target_batches[:,t] 
            else:
                # change decoder
                decoder_input = topvi.squeeze()
                # top_ptr_i = torch.gather(story[:, :, 0], 1, Variable(toppi))
                # next_in = [top_ptr_i[i].item() if (toppi[i].item() < story_lengths[i] - 1) else topvi[i].item() for i in range(batch_size)]
                # decoder_input = Variable(torch.LongTensor(next_in))
            
            if get_decoded_words:
        #         temp = []
        #         from_which = []
        #         for bi in range(batch_size):
        #             if (toppi[bi].item() < story_lengths[bi] - 1):
        #                 temp.append(copy_list[bi][toppi[bi].item()])
        #                 from_which.append('p')
        #             else:
        #                 ind = topvi[bi].item()
        #                 if ind == EOS_token:
        #                     temp.append('EOS')
        #                 else:
        #                     temp.append(self.lang.index2word[ind])
        #                 from_which.append('v')
        #         decoded_words.append(temp)
        #         self.from_whichs.append(from_which)
        # self.from_whichs = np.array(self.from_whichs)
        # decoded_words = np.array(decoded_words)
        # decoded_words = decoded_words.transpose()
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:,i][bi] < story_lengths[bi]-1:
                                cw = copy_list[bi][toppi[:,i][bi].item()]
                                break
                        temp_f.append(cw)

                        if args['record']:
                            memory_mask_for_step[bi, toppi[:,i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse, all_decoder_outputs_gate_signal, decoded_words

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_



class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
