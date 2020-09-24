import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.dropout_layer = nn.Dropout(dropout)
        self.a1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(output_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(output_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(2)

    def forward(self, input, adj):  # input: batch_size * max_len * embedding_dim, adj: batch_size * max_len * max_len.
        # add for mimic memory
        h = input
        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(1,2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime