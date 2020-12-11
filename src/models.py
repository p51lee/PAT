from layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FCGraphAttentionLayer

class FCGAT(nn.Module):
    def __init__(self, n_input_features, n_hidden_features1, n_hidden_features2, n_output_features, dropout, alpha, n_heads1, n_heads2, num_particle):
        super(FCGAT, self).__init__()
        self.dropout = dropout
        self.N = num_particle

        self.attentions = [FCGraphAttentionLayer(n_input_features, n_hidden_features1, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.middle_attentions = [FCGraphAttentionLayer(n_hidden_features1 * n_heads1, n_hidden_features2, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(n_heads2)]
        # for j, attention in enumerate(self.middle_attentions):
        #     self.add_module('attention_middle_{}'.format(j), attention)

        self.out_att = FCGraphAttentionLayer(n_hidden_features1 * n_heads1, n_output_features, dropout=dropout, alpha=alpha, concat=False)

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