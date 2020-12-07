from layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FCGraphAttentionLayer

class FCGAT(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features, dropout, alpha, n_heads):
        super(FCGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [FCGraphAttentionLayer(n_input_features, n_hidden_features, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = FCGraphAttentionLayer(n_hidden_features * n_heads, n_output_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        return F.dropout(x, self.dropout, training=self.training)