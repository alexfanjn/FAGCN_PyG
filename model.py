import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree



class FALayer(MessagePassing):
    def __init__(self, data, num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')
        self.data = data
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.row, self.col = data.edge_index
        self.norm_degree = degree(self.row, num_nodes=data.y.shape[0]).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h):
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
        norm = self.dropout(norm)
        return self.propagate(self.data.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out


class FAGCN(nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(data, num_hidden, dropout))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)