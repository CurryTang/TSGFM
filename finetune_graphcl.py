import argparse
from graphcl.loader import MoleculeDataset
from data.chemmol.gen_data import MolOFADataset
from utils import SentenceEncoder
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from graphcl.model import GNN, GNN_graphpred, TextGNN, TextGNN_pred
from sklearn.metrics import roc_auc_score
from graphcl.splitters import scaffold_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
import warnings
from ogb.graphproppred import Evaluator
warnings.filterwarnings("ignore")

criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion2 = nn.BCEWithLogitsLoss()
def train(args, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        # import ipdb; ipdb.set_trace()
        if args.dataset == 'pcba' or args.dataset == 'hiv':
            is_labeled = batch.y == batch.y
            loss = criterion2(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            optimizer.zero_grad()
        else:
            y = batch.y.view(pred.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
        # import ipdb; ipdb.set_trace()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        # import ipdb; ipdb.set_trace()
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    # import ipdb; ipdb.set_trace()
    if args.dataset == 'pcba':
        evaluator =  Evaluator(name='ogbg-molpcba')
        input_dict = {"y_true": y_true, "y_pred": y_scores}
        return evaluator.eval(input_dict)['ap']
    elif args.dataset == 'hiv':
        evaluator =  Evaluator(name='ogbg-molhiv')
        input_dict = {"y_true": y_true, "y_pred": y_scores}
        return evaluator.eval(input_dict)['rocauc']
    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
        try:
            return sum(roc_list)/len(roc_list) #y_true.shape[1]
        except ZeroDivisionError:
            return 0



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--space', type=str, default = 'element', help='space of the dataset. graph or text')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    if args.space == 'element':
        dataset = MoleculeDataset("dataset/dataset/" + args.dataset, dataset=args.dataset)
        # import ipdb; ipdb.set_trace()
    else:
        encoder = SentenceEncoder("minilm", "/localscratch/chenzh85", batch_size=256)
        dataset = MolOFADataset(name=args.dataset, encoder=encoder, root="cache_data_minilm", load_text=True)
        # import ipdb; ipdb.set_trace()
    print(dataset)

    if args.dataset == 'pcba' or args.dataset == 'hiv':
        train_dataset = dataset[dataset.splits['train']]
        valid_dataset = dataset[dataset.splits['valid']]
        test_dataset = dataset[dataset.splits['test']]
    else:        
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
            # import ipdb; ipdb.set_trace()
        else:
            raise ValueError("Invalid split option.")

    print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    if args.space == 'element':
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    else:
        setattr(args, 'emb_dim', 384)
        model = TextGNN_pred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        print("")

    with open('result.log', 'a+') as f:
        f.write(args.dataset + ' ' + str(args.runseed) + ' ' + str(np.array(test_acc_list)[-1]))
        f.write('\n')

if __name__ == "__main__":
    main()