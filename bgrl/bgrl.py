import copy

import torch
from graphmae.utils import mask_edge, create_norm
from graphmae.models.gcn import GCN
import dgl
import numpy as np

def drop_features(g, p):
    features = g.ndata['x']
    drop_mask = torch.empty(
        (features.size(1),), dtype=torch.float32, device=features.device
    ).uniform_(0, 1) < p
    features[:, drop_mask] = 0
    g.ndata['x'] = features
    return g

def drop_edges(g, p):
    if p <= 0:
        return g 
    g = g.remove_self_loop()
    mask_index = mask_edge(g, p)
    g = dgl.remove_edges(g, mask_index).add_self_loop()
    return g

def drop_feature_edges(g, p_f, p_e):
    g = drop_features(g, p_f)
    g = drop_edges(g, p_e)
    return g

class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        inputs = online_x.ndata['x']
        online_y = self.online_encoder(online_x, inputs)

        # prediction
        online_q = self.predictor(online_y)

        inputs = target_x.ndata['x']
        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x, inputs).detach()
        return online_q, target_y


def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]

from torch import nn

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()



def build_bgrl_model(args):
    gnn_model = GCN(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        out_dim=args.num_hidden if args.num_out == -1 else args.num_out,
        num_layers=args.num_layers,
        dropout=args.in_drop,
        activation=args.activation,
        residual=args.residual,
        norm=create_norm(args.norm),
        encoding=False 
    )

    predictor = MLP_Predictor(
        input_size=args.num_hidden if args.num_out == -1 else args.num_out,
        output_size=args.num_hidden if args.num_out == -1 else args.num_out,
        hidden_size=args.predictor_hidden_size
    )

    model = BGRL(gnn_model, predictor)
    return model


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))