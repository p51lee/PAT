import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(FCGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # print(h.size(), self.W.size())
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 스칼라가 되어버린 맨 마지막 차원을 없앤다?

        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(nn.Module):  # FCGAT와 동일, 하지만 새로운 layer로서의  역할
    def __init__(self, n_input_features, n_hidden_features, n_output_features, dropout, alpha, n_heads, num_particle):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.N = num_particle

        self.attentions = [FCGraphAttentionLayer(n_input_features, n_hidden_features, dropout=dropout, alpha=alpha) for
                           _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = FCGraphAttentionLayer(n_hidden_features * n_heads, n_output_features, dropout=dropout,
                                             alpha=alpha)

        self.W = nn.Parameter(torch.empty(size=(1, self.N)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att_mid(x) for att_mid in self.middle_attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x)

        x = torch.mm(self.W, x).squeeze(0)
        # print(x.size())
        return x


class PATLayer(nn.Module):  # FCGAT와 동일한 input을 받아서 여러 개 particle 차원의 output
    def __init__(self, num_particles, dimension, n_hidden_features, dropout, alpha, n_heads, name="PAT layer"):
        super(PATLayer, self).__init__()

        self.num_particles = num_particles
        self.dim = dimension
        self.name = name

        self.GAT_x = GATLayer(
            n_input_features=dimension * 2,
            n_hidden_features=n_hidden_features,
            n_output_features=dimension * 2,
            dropout=dropout,
            n_heads=n_heads,
            alpha=alpha,
            num_particle=num_particles
        )

        self.add_module('GAT_x', self.GAT_x)

        self.W_relativity = nn.Parameter(torch.empty(size=(num_particles, num_particles)))
        nn.init.xavier_uniform_(self.W_relativity.data, gain=1.414)

        self.W_weightSum = nn.Parameter(torch.empty(size=(1, num_particles)))
        nn.init.xavier_uniform_(self.W_weightSum.data, gain=1.414)

    def forward(self, init_frame):
        x_out_list = []

        for ptl_idx in range(self.num_particles):
            init_frame_indexed = torch.cat((init_frame[ptl_idx:], init_frame[:ptl_idx]), 0)

            if self.training:
                init_frame_indexed = init_frame_indexed.cuda()

            # W matrix를 곱해줌으로서 첫 번째  particle이 특별하다는 것을 학습

            # print(self.W_relativity.size())
            # print(init_frame)
            # print(self.name)

            init_frame_relative = torch.mm(self.W_relativity, init_frame_indexed)
            x = self.GAT_x(init_frame_relative)
            # x = torch.mm(self.W_weightSum, x).squeeze(0)

            x_out_list.append(x)

        return torch.stack(x_out_list, 0)
