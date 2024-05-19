import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from graphmae.utils import build_args
from graphmae.data_util import unify_dataset_loader
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import mask_to_index
from torch_geometric import seed_everything


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def mask2splitidx(masks):
    split_idx = {}
    split_idx['train'] = mask_to_index(masks[0])
    split_idx['valid'] = mask_to_index(masks[1])
    split_idx['test'] = mask_to_index(masks[2])
    return split_idx


def main():
    ## this file is mainly designed for single-task baseline
    args = build_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = unify_dataset_loader([args.dataset], args)[0]
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    data.y = data.y.view(-1, 1)

    split_idx = mask2splitidx([dataset.train_mask, dataset.val_mask, dataset.test_mask])

    train_idx = split_idx['train'].to(device)
    num_features = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    if args.encoder == 'sage':
        model = SAGE(num_features, args.num_hidden,
                     num_classes, args.num_layers,
                     args.in_drop).to(device)
    else:
        model = GCN(num_features, args.num_hidden,
                    num_classes, args.num_layers,
                    args.in_drop).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    # logger = Logger(args.runs, args)

    best_val = 0
    best_test = 0
    seed_everything(0)
    for run in range(1):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, 1 + args.max_epoch):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            # logger.add_result(run, result)

            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
            if valid_acc > best_val:
                best_val = valid_acc
                best_test = test_acc
        print("Best Valid: ", best_val, "Best Test: ", best_test)

        #logger.print_statistics(run)
    # logger.print_statistics()


if __name__ == "__main__":
    main()