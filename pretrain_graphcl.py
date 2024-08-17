import argparse
from graphcl.loader import MoleculeDataset_aug
from data.chemmol.gen_data import MolOFADataset_aug, MolOFADataset
from utils import SentenceEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from graphcl.model import GNN, TextGNN
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn, dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        # import ipdb; ipdb.set_trace()
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, datasets, optimizer, space='element'):
    print("Training GraphCL")
    train_accs = []
    train_losses = []
    for i, dataset in enumerate(datasets):
        dataset.aug = "none"
        dataset1 = dataset
        dataset2 = deepcopy(dataset1)
        dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
        dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

        loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
        loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

        model.train()

        train_acc_accum = 0
        train_loss_accum = 0

        for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            # import ipdb; ipdb.set_trace()
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            loss = model.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
            acc = torch.tensor(0)
            train_acc_accum += float(acc.detach().cpu().item())

        train_acc = train_acc_accum/(step+1) 
        train_loss = train_loss_accum/(step+1)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        print(f"Dataset {i} Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
    return np.mean(train_accs), np.mean(train_losses)


def set_up_pretraining_dataset(dataset_str, space):
    data_names = dataset_str.split(",")
    datasets = []
    for datan in data_names:
        if space == 'text':
            encoder = SentenceEncoder("minilm", "/localscratch/chenzh85", batch_size=256)
            dataset = MolOFADataset_aug(name=datan, encoder=encoder, root="cache_data_minilm", load_text=True)
            # import ipdb; ipdb.set_trace()
        else:
            dataset = MoleculeDataset_aug("dataset/dataset/" + datan, dataset=datan)
            # import ipdb; ipdb.set_trace()
        datasets.append(dataset)
    # import ipdb; ipdb.set_trace()
    return datasets



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    parser.add_argument('--space', type=str, default = 'element', choices=['element', 'text'])
    parser.add_argument('--save_epoch', type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #set up dataset
    # dataset = MoleculeDataset_aug("dataset/dataset/" + args.dataset, dataset=args.dataset)
    # print(dataset)
    datasets = set_up_pretraining_dataset(args.dataset, args.space)
    #set up mode
    if args.space == 'element':
        gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    else:
    #     ## -1 is designed for pre-training
        setattr(args, 'emb_dim', 384)
    #     # import ipdb; ipdb.set_trace()
        gnn = TextGNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio)
    model = graphcl(gnn, args.emb_dim)
    model.to(device)
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))
        train(args, model, device, datasets, optimizer, space=args.space)
        if epoch % args.save_epoch == 0:
            print("====saving")
            torch.save(gnn.state_dict(), f"./models_graphcl/graphcl_{args.dataset}_{args.space}" + str(epoch) + ".pth")

if __name__ == "__main__":
    main()