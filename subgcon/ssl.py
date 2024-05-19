## SUPPORTED SSL METHODS: DGI, GCC, BGRL

import torch
import GCL
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

class DGIEncoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(DGIEncoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index, batch=None):
        z = self.encoder(x, edge_index)

        if batch is None:
            g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        else:
            g = self.project(torch.sigmoid(global_mean_pool(z, batch)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn

def dgi_train_step(encoder_model, contrast_model, data, optimizer):
    # model.train()
    # optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn, batch=data.batch)
    return loss