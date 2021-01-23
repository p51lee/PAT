from layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FCGraphAttentionLayer


class FCGAT(nn.Module):
    def __init__(self, n_input_features, n_hidden_features1, n_hidden_features2, n_output_features, dropout, alpha,
                 n_heads1, n_heads2, num_particle):
        super(FCGAT, self).__init__()
        self.dropout = dropout
        self.N = num_particle

        self.attentions = [FCGraphAttentionLayer(n_input_features, n_hidden_features1, dropout=dropout, alpha=alpha) for
                           _ in
                           range(n_heads1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.middle_attentions = [FCGraphAttentionLayer(n_hidden_features1 * n_heads1, n_hidden_features2, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(n_heads2)]
        # for j, attention in enumerate(self.middle_attentions):
        #     self.add_module('attention_middle_{}'.format(j), attention)

        self.out_att = FCGraphAttentionLayer(n_hidden_features1 * n_heads1, n_output_features, dropout=dropout,
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


class RPAT(nn.Module):
    def __init__(self, num_particles, dimension, n_hidden_features, dropout, alpha, n_heads):
        super(RPAT, self).__init__()

        self.PAT_x = PATLayer(
            n_hidden_features=n_hidden_features,
            dimension=dimension,
            dropout=dropout,
            n_heads=n_heads,
            alpha=alpha,
            num_particles=num_particles,
            name="PAT_x"
        )

        self.PAT_h = PATLayer(
            n_hidden_features=n_hidden_features,
            dimension=dimension,
            dropout=dropout,
            n_heads=n_heads,
            alpha=alpha,
            num_particles=num_particles,
            name="PAT_h"
        )

        self.PAT_y = PATLayer(
            n_hidden_features=n_hidden_features,
            dimension=dimension,
            dropout=dropout,
            n_heads=n_heads,
            alpha=alpha,
            num_particles=num_particles,
            name="PAT_y"
        )

        self.add_module('PAT_x', self.PAT_x)
        self.add_module('PAT_h', self.PAT_h)
        self.add_module('PAT_y', self.PAT_y)

        # physical symmetry 때문에 0으로 초기화
        self.h_0 = torch.zeros(size=(num_particles, dimension * 2))

    def forward(self, states):
        # print("states", states)
        output_list = []
        for i, state in enumerate(states):
            if i == 0:
                h_t = self.get_h_t(state, self.h_0)
            else:
                h_t = self.get_h_t(state, h_t)

            # output_list.append(self.get_y_t(h_t))  # 이거는 절대적인 좌표를 학습하는 방식
            output_list.append(state + self.get_y_t(h_t))  # 이거는 상대적인 차이를 학습하는 방식
            # output_list.append(state)  # 이거는 대조군,  train.py 의 loss_train.backward() 과
            #     optimizer.step() 을 주석처리해놓으면 한 characteristic 당 0.0523 의 오차가 생김을 볼 수 있다.
            # 따라서 이거보다 큰 오차를 가지면 큰일난다.

        return torch.stack(output_list)  # 다다다 출력한 y값들이 나가서 학습에 사용된다.

    def get_h_t(self, x_t, h_tm1):  # h_{t-1}
        Px = self.PAT_x(x_t)  # 입력 x에 pat를 적용한 것
        Ph = self.PAT_h(h_tm1)  # 그 전의 hidden state에 pat 적용한 것
        return Px + Ph

    def get_y_t(self, h_t):
        return self.PAT_y(h_t)


class RPATRecursive(RPAT):  # 이 클래스는 input 을 한 번만 받고 계속 자신의 결과를 이용하면서 운동을 예측한다.
    def __init__(self, num_particles, dimension, n_hidden_features, dropout, alpha, n_heads):
        super(RPATRecursive, self).__init__(num_particles, dimension, n_hidden_features, dropout, alpha, n_heads)

    def forward(self, states):
        state = states[0]
        output_list = []
        for i, _ in enumerate(states):
            if i == 0:
                h_t = self.get_h_t(state, self.h_0)
            else:
                h_t = self.get_h_t(state, h_t)

            state = state + self.get_y_t(h_t)
            output_list.append(state)  # 이거는 상대적인 차이를 학습하는 방식

        return torch.stack(output_list)  # 다다다 출력한 y값들이 나가서 학습에 사용된다.
