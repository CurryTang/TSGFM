#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200 
import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from subgcon.efficient_dataset import NodeSubgraphDataset, LinkSubgraphDataset, GraphSubgraphDataset
from subgcon.utils_mp import Subgraph, preprocess
from subgcon.models import SugbCon, Encoder, Scorer, Pool
from graphmae.utils import build_args, create_optimizer, set_random_seed, load_best_configs
from graphmae.data_util import unify_dataset_loader
from graphmae.models import build_model, TaskHeadModel
import logging
import os.path as osp
from tqdm import tqdm
from graphmae.config import DATASET

def train(model, optimizer, loader, args, scheduler, device, model_type='subgcon', mask = None, head_name = None):
    # Model training
    model.train()
    optimizer.zero_grad()
    avg_loss = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    for batch in loader:
        index = batch.ptr[:-1]
        if model_type == 'subgcon':
            z, summary = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device))
            loss = model.loss(z, summary)
        elif model_type == 'graphmae':
            loss, loss_item = model(batch.x.to(device), batch.edge_index.to(device))
        elif model_type == 'cotrain':
            ## mask and head_name should be provided for this case
            output = model(batch.x.to(device), batch.edge_index.to(device), head_name)
            loss = ce_loss(output[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        avg_loss += loss.item()
    return loss.item() / len(loader)

def get_all_node_emb(model, num_node, loader, args, device, level='node'):
    # Obtain central node embs from subgraphs 
    model.eval()
    z = []
    for batch in loader:
        index = batch.ptr
        node, graph = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device))
        if level == 'graph':
            z.append(graph)
        else:
            z.append(node[batch.ptr[:-1]])
    z = torch.cat(z, dim = 0)
    return z
    
    
def node_level_test(model, dataset, loader, args, device):
    # Model testing
    model.eval()
    num_node = dataset.num_nodes
    with torch.no_grad():
        all_emb = get_all_node_emb(model, num_node, loader, args, device, level = 'node')
        train_z = all_emb[data.train_mask]
        val_z = all_emb[data.val_mask]
        test_z = all_emb[data.test_mask]
    
    train_y = data.y[data.train_mask]
    val_y = data.y[data.val_mask]
    test_y = data.y[data.test_mask]
    val_acc, test_acc = model.test(train_z, train_y, val_z, val_y, test_z, test_y)
    print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
    return val_acc, test_acc

def link_level_test(model, dataset, loader, args, device):
    pass 

def graph_level_test(model, dataset, loader, args, device):
    pass


def test(model, dataset, loader, args, device, level='node'):
    if level == 'node':
        return node_level_test(model, dataset, loader, args, device)
    elif level == 'link':
        return link_level_test(model, dataset, loader, args, device)
    elif level == 'graph':
        return graph_level_test(model, dataset, loader, args, device)


def main(args):
    ## load necessary args for all methods
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    encoder = args.encoder
    decoder = args.decoder
    num_hidden = args.num_hidden
    drop_edge_rate = args.drop_edge_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    logs = args.logging
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    feature_dim = args.feature_dim
    args.num_features = feature_dim 
    ## pretrain part
    max_epochs = args.max_epoch
    use_scheduler = args.scheduler
    ## select method
    if args.method == 'subgcon':
        model = SugbCon(
            hidden_channels=args.num_hidden, encoder=Encoder(feature_dim, args.num_hidden),
            pool=Pool(in_channels=args.num_hidden),
            scorer=Scorer(args.num_hidden)).to(device)
    elif args.method == 'graphmae':
        model = build_model(args)
    elif args.method == 'cotrain':
        model = build_model(args)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if use_scheduler:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epochs) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    
    ## load datas
    datas = unify_dataset_loader(args.pre_train_datasets, args)
    ## saved stats
    best_avg_val = 0
    val_accs = []
    test_accs = []
    

    for e in range(max_epochs):
        ## pre-train stage
        for i, G in enumerate(datas):
            train_loader = G.search(batch_size = batch_size, shuffle = True)
            loss = train(model, optimizer, train_loader, args, scheduler, device)
            logging.info("Epoch {} Dataset {} Loss {}".format(e, args.pre_train_datasets[i], loss))
        ## evaluation part
        eval_val = []
        eval_test = []
        for i, G in enumerate(datas):
            test_loader = G.search(batch_size = batch_size, shuffle = False)

            val_res, test_res = test(model, G, test_loader, args, device, model_type = args.method, mask = G.train_mask, head_name = args.pre_train_datasets[i])
            eval_val.append(val_res)
            eval_test.append(test_res)
        for i in range(len(graphs)):
            logging.info("Epoch {} Dataset {} Val-{} {} Test-{} {}".format(e, args.downstream_datasets[i], DATASET[args.downstream_datasets[i]]['metric'], eval_val[i], DATASET[args.downstream_datasets[i]]['metric'], eval_test[i]))
        avg_val = np.mean(eval_val)
        if avg_val > best_avg_val:
            best_avg_val = avg_val
            val_accs = eval_val
            test_accs = eval_test
    
    logging.info("Best Avg Val {}".format(best_avg_val))
    for i in range(len(graphs)):
        logging.info("Dataset: {} Best Val {} Test {}".format(args.pre_train_datasets[i], val_accs[i], test_accs[i]))
    return best_avg_val, val_accs, test_accs    
            





    

if __name__ == '__main__':
    args = build_args()
    print(args)
    if not args.not_same_pretrain_downstream:
        args.downstream_datasets = args.pre_train_datasets
    main(args)
    

  
        
        
    

    
