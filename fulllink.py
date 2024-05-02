import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from graphmae.data_util import unify_dataset_loader
from graphmae.utils import build_args
from torch_geometric.transforms import ToSparseTensor
import numpy as np

def keep_attrs_for_data(data):
    for k in data.keys():
        if k not in ['x', 'edge_index', ]:
            data[k] = None
    return data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, dataset, optimizer, batch_size, device):
    model.train()
    predictor.train()

    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = dataset.train_edge_index.t().to(device)
    data = dataset._data.to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, dataset, evaluator, batch_size, device):
    model.eval()
    predictor.eval()

    data = dataset._data.to(device)
    h = model(data.x, data.adj_t)

    # pos_train_edge = split_edge['train']['edge'].to(h.device)
    # pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    # neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    # pos_test_edge = split_edge['test']['edge'].to(h.device)
    # neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_edge = dataset.train_edge_index.t().to(device)
    pos_valid_edge = dataset.pos_val_edge_index.t().to(device)
    neg_valid_edge = dataset.neg_val_edge_index.t().to(device)
    pos_test_edge = dataset.pos_test_edge_index.t().to(device)
    neg_test_edge = dataset.neg_test_edge_index.t().to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.adj_t)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
        print(f'Hits@{K}: train hits {train_hits:.4f}, valid hits {valid_hits:.4f}, test_hits {test_hits:.4f}')
    return results



def train_over_multiple_datasets(model, predictor, datasets, evaluator, args, device):
    model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr)
    avg_val = 0
    avg_test = 0
    best_vals = []
    best_tests = []
    best_epoch = 0
    for epoch in range(1, 1 + args.max_epoch):
        for dataset in datasets:
            loss = train(model, predictor, dataset, optimizer,
                         args.batch_size, device)
            print(f"Dataset: {dataset.name}, Epoch: {epoch}, Loss: {loss:.4f}")
        
        curr_val = []
        curr_test = []
        ## after train, iterate over datasets to test
        for dataset in datasets:
            results = test(model, predictor, dataset, evaluator,
                           args.batch_size, device)
            hits100 = results['Hits@100']
            curr_val.append(hits100[1])
            curr_test.append(hits100[2])
        tmp_avg_val = np.mean(curr_val)
        tmp_avg_test = np.mean(curr_test)
        print(f"Epoch: {epoch}, Avg Val Hits@100: {tmp_avg_val:.4f}, Avg Test Hits@100: {tmp_avg_test:.4f}")
        if tmp_avg_val > avg_val:
            avg_val = tmp_avg_val
            avg_test = tmp_avg_test
            best_vals = curr_val
            best_tests = curr_test
            best_epoch = epoch
    print(f"Best Avg Val Hits@100: {avg_val:.4f}, Best Avg Test Hits@100: {avg_test:.4f}")
    print(f"Best Val Hits@100: {best_vals}")
    print(f"Best Test Hits@100: {best_tests}")
    print(f"Best Epoch: {best_epoch}")
    return avg_val
    



def main():
    args = build_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    args.subgraph_size = -1
    datasets = unify_dataset_loader(args.pre_train_datasets, args)
    # dataset = PygLinkPropPredDataset(name='ogbl-collab')
    
    ## train stage
    model = GCN(args.feature_dim, args.num_hidden,
        args.num_hidden, args.num_layers,
        args.in_drop).to(device)

    predictor = LinkPredictor(args.num_hidden, args.num_hidden, 1,
                              args.num_layers, args.in_drop).to(device)

    evaluator = Evaluator(name='ogbl-collab')

    for d in datasets:
        d._data = keep_attrs_for_data(d._data)
        d._data = ToSparseTensor(remove_edge_index=False)(d._data)
    
    train_over_multiple_datasets(model, predictor, datasets, evaluator, args, device)

    

        



if __name__ == "__main__":
    main()