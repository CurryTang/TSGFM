import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation

class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 norm,
                 encoding=False
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.mlp_layers.append(MLPLayer(
                in_dim, out_dim, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.mlp_layers.append(MLPLayer(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                self.mlp_layers.append(MLPLayer(
                    num_hidden, num_hidden, norm=norm, activation=create_activation(activation)))
            # output projection
            self.mlp_layers.append(MLPLayer(
                num_hidden, out_dim, activation=last_activation, norm=last_norm))
        self.norms = None
        self.head = nn.Identity()

    def forward(self, x, edge_index, return_hidden=False):
        h = x
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.mlp_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)

    def reset_back(self):
        self.head = nn.Identity()



class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)        
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x):
        rst = self.fc(x)
        if self.norm is not None:
            rst = self.norm(rst)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst