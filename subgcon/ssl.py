## SUPPORTED SSL METHODS: DGI, GCC, BGRL

import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
import copy

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


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batchnorm'):
        super().__init__()
        if norm is None:
            self.norm = lambda x: x
        elif dim is None or norm.lower() == 'none':
            self.norm = lambda x: x
        elif norm.lower() == 'batchnorm':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm.lower() == 'layernorm':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class BGRLEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(BGRLEncoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.batch_norm = Normalize(hidden_dim, norm=predictor_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        h1 = self.online_encoder(x1, edge_index1, edge_weight1)
        h1 = self.batch_norm(h1)
        h1_online = self.projection_head(h1)
        h2 = self.online_encoder(x2, edge_index2, edge_weight2)
        h2 = self.batch_norm(h2)
        h2_online = self.projection_head(h2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            h1 = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            h1 = self.batch_norm(h1)
            h1_target = self.projection_head(h1)
            h2 = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            h2 = self.batch_norm(h2)
            h2_target = self.projection_head(h2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

def bgrl_train_step(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    # loss.backward()
    # optimizer.step()
    # encoder_model.update_target_encoder(0.99)
    return loss 