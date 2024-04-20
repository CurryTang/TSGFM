from .edcoder import PreModel
from .gat import GAT
from torch_geometric.nn.models import GCN
from .mlp import MLP
import torch
from ..config import DATASET as task_config

def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features


    model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model

def build_model_backbone(args, in_dim, out_dim):
    ## for supervised baseline, we only have encoder
    ## encoding should be False
    if args.encoder == 'gat':
        return GAT(
            in_dim, 
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=create_norm(args.norm),
            concat_out=True,
            encoding=False
        ) 
    elif args.encoder == 'gcn':
        return GCN(
            in_dim,
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            dropout=args.in_drop,
            activation=args.activation,
            residual=args.residual,
            norm=create_norm(args.norm),
            encoding=False
        )
    elif args.encoder == 'mlp':
        return MLP(
            in_dim,
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            dropout=args.in_drop,
            activation=args.activation,
            norm=create_norm(args.norm),
            encoding=False
        )


class TaskHeadModel(torch.nn.Module):
    def __init__(self, args, in_dim, encoder_space_dim, task_config):
        super(MultiheadModel, self).__init__()
        self.encoder = build_model_backbone(args, in_dim, encoder_space_dim)
        self.heads = {}
        self.Gs = []
        for d in args.pre_train_datasets:
            dim = task_config[d]['task_dim']
            t_head = torch.nn.Linear(encoder_space_dim, dim)
            self.heads[d] = t_head
        
    
    def forward(self, x, edge_index, head_name):
        x = self.encoder(x, edge_index)
        x = self.heads[head_name](x)
        return x