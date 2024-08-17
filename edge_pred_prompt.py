import argparse

from graphcl.loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from graphcl.model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_mean_pool
from graphcl.splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

class Gprompt(torch.nn.Module):
    def __init__(self,input_dim):
        super(Gprompt, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embeddings):
        node_embeddings=node_embeddings*self.weight
        return node_embeddings

def group_embeddings_by_multilabel(labels, embeddings):
    """
    Groups embeddings based on multi-labels, separating embeddings for 0 and 1 labels.

    Args:
        labels: A tensor of shape (N, k) representing multi-labels.
        embeddings: A tensor of shape (N, d) representing the embeddings.

    Returns:
        A tensor of shape (2, k, d) representing the grouped embeddings,
        where the first dimension corresponds to label 0/1,
        the second dimension corresponds to the classes, and
        the third dimension corresponds to the embedding dimension.
    """

    N, k = labels.shape
    _, d = embeddings.shape

    # Create index tensors for scatter_add for labels 0 and 1
    indices_1 = labels.nonzero().t()[0]  # Indices where labels are 1
    indices_0 = (labels == 0).nonzero().t()[0] # Indices where labels are 0

    # Initialize a tensor to store the grouped embeddings
    grouped_embeddings = torch.zeros(2, k, d, dtype=embeddings.dtype, device=embeddings.device)

    # Use scatter_add to group embeddings for label 1
    grouped_embeddings[1].scatter_add_(0, indices_1.unsqueeze(1).expand(-1, d), embeddings)

    # Use scatter_add to group embeddings for label 0
    grouped_embeddings[0].scatter_add_(0, indices_0.unsqueeze(1).expand(-1, d), embeddings)

    return grouped_embeddings


def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts


class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Gprompt_tuning_loss, self).__init__()
        self.tau = tau
    
    def forward(self, embedding, center_embedding, labels):
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1) / self.tau
        exp_similarities = torch.exp(similarity_matrix)
        # Sum exponentiated similarities for the denominator
        pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
        # select the exponentiated similarities for the correct classes for the every pair (xi,yi)
        pos = exp_similarities.gather(1, labels.view(-1, 1))
        L_prompt = -torch.log(pos / pos_neg)
        loss = torch.sum(L_prompt)
                    
        return loss


def prompt_train(args, model, device, loader, optimizer, prompt, class_num):
    accumulated_centers = []
    accumulated_counts = [] 
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
        # import ipdb; ipdb.set_trace()
        for i in range(batch.y.shape[1]):
            center, class_counts = center_embedding(out, batch.y[i], 2)
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item()
    mean_centers = accumulated_centers / accumulated_counts 
    return total_loss / len(loader), mean_centers

criterion = nn.BCEWithLogitsLoss(reduction = "none")
def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def prompt_eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            similarity_matrix = F.cosine_similarity(pred.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            pred = similarity_matrix.argmax(dim=1)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    return roc_auc_score(y_true, y_scores)
            

def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
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
    dataset = MoleculeDataset("dataset/dataset/" + args.dataset, dataset=args.dataset)

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    # model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    #if not args.input_model_file == "":
        #model.from_pretrained(args.input_model_file)
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model.load_state_dict(torch.load(args.input_model_file))
    prompt = Gprompt(args.emb_dim)
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    # model_param_group = []
    # model_param_group.append({"params": model.gnn.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(prompt.parameters(), lr=args.lr, weight_decay=args.decay)
    # print(optimizer)
    

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        prompt_train(args, model, device, train_loader, optimizer, prompt, num_tasks)
        val_acc = prompt_eval(args, model, device, val_loader)
        test_acc = prompt_eval(args, model, device, test_loader)
        train_acc = 0
        # train(args, model, device, train_loader, optimizer)

        # print("====Evaluation")
        # if args.eval_train:
        #     train_acc = eval(args, model, device, train_loader)
        # else:
        #     print("omit the training accuracy computation")
        #     train_acc = 0
        # val_acc = eval(args, model, device, val_loader)
        # test_acc = eval(args, model, device, test_loader)

        # print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()