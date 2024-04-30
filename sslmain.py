#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200 
import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool
from subgcon.efficient_dataset import NodeSubgraphDataset, LinkSubgraphDataset, GraphSubgraphDataset
from subgcon.utils_mp import Subgraph, preprocess
from subgcon.models import SugbCon, Encoder, Scorer, Pool
from graphmae.utils import build_args, create_optimizer, set_random_seed, load_best_configs
from graphmae.data_util import unify_dataset_loader
from graphmae.models import build_model, TaskHeadModel
from graphmae.evaluation import linear_test, link_linear_test, linear_mini_batch_test
from graphmae.models.gcn import GNN_node
import logging
import os.path as osp
from tqdm import tqdm
from graphmae.config import DATASET
from ogb.linkproppred.evaluate import Evaluator
import pytorch_warmup as warmup

def train(model, optimizer, loader, args, scheduler, device, model_type='subgcon', mask = None, head_name = None, warmup_scheduler = None):
    # Model training
    model.train()
    optimizer.zero_grad()
    avg_loss = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    for batch in loader:
        if batch.x.dim() == 1 and args.embed_mode == 'llm':
            batch.x = batch.node_text_feat
            batch.xe = batch.edge_text_feat
        if model_type == 'subgcon':
            index = batch.ptr[:-1]
            edge_feat = batch.xe if hasattr(batch, 'xe') else None
            z, summary = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device), edge_feat)
            loss = model.loss(z, summary)
        elif model_type == 'graphmae':
            loss, loss_item = model(batch.x.to(device), batch.edge_index.to(device))
        elif model_type == 'cotrain':
            ## mask and head_name should be provided for this case
            output = model(batch.x.to(device), batch.edge_index.to(device), head_name)
            loss = ce_loss(output[mask], batch.y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if scheduler is not None:
        if warmup_scheduler is None:
            scheduler.step()
        else:
            with warmup_scheduler.dampening():
                scheduler.step()
    avg_loss += loss.item()
    return avg_loss / len(loader)

def get_all_node_emb(model, num_node, loader, args, device, level='node', modeln='subgcon'):
    # Obtain central node embs from subgraphs 
    model.eval()
    z = []
    size = len(loader)
    for batch in loader:
        if batch.x.dim() == 1 and args.embed_mode == 'llm':
            batch.x = batch.node_text_feat
            batch.xe = batch.edge_text_feat
        index = batch.ptr[:-1]
        if modeln == 'subgcon':
            edge_feat = batch.xe if hasattr(batch, 'xe') else None
            if edge_feat is not None:
                node, graph = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device), edge_feat.to(device))
            else:
                node, graph = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device))
        elif modeln == 'graphmae':
            # import ipdb; ipdb.set_trace()
            out = model.embed(batch.x.to(device), batch.edge_index.to(device))
            if size > 1:
                node = out[index]
            else:
                node = out
            graph = global_mean_pool(out, batch.batch.to(device))
            # import ipdb; ipdb.set_trace()
        if level == 'graph':
            z.append(graph)
        else:
            z.append(node)
    z = torch.cat(z, dim = 0)
    return z
    
    
def node_level_test(model, dataset, loader, args, device):
    # Model testing
    model.eval()
    num_node = dataset.num_nodes
    with torch.no_grad():
        all_emb = get_all_node_emb(model, num_node, loader, args, device, level = 'node', modeln=args.method)

    dataset.data.num_classes = dataset.data.y.max().item() + 1
    if args.cpuinf:
        test_acc, estp_test_acc, val_acc = linear_test(all_emb, dataset, 100, device, eval_device='cpu')
    else:
        test_acc, estp_test_acc, val_acc = linear_mini_batch_test(all_emb, dataset, 100, device, m_name = DATASET[dataset.name]['metric'])
    
    # val_acc, test_acc = model.test(train_z, train_y, val_z, val_y, test_z, test_y)
    print('val_acc = {} test_acc = {}'.format(val_acc, estp_test_acc))
    return val_acc, test_acc

def link_level_test(model, dataset, loader, args, device):
    model.eval()
    with torch.no_grad():
        z = get_all_node_emb(model, 1, loader, args, device, level = 'link', modeln=args.method)
    
    evaluator = Evaluator(name='ogbl-ppa')
    dataset.data.num_classes = dataset.data.y.max().item() + 1
    test_acc, val_acc = link_linear_test(z, dataset, 100, device, evaluator)
    print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
    return val_acc, test_acc 

def graph_level_test(model, dataset, loader, args, device):
    model.eval()
    with torch.no_grad():
        z = get_all_node_emb(model, 1, loader, args, device, level = 'graph', modeln=args.method)
    
    test_acc, estp_test_acc, val_acc = linear_mini_batch_test(z, dataset, 100, device, m_name = DATASET[dataset.name]['metric'])
    print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
    return val_acc, test_acc


def test(model, dataset, loader, args, device, level='node'):
    if level == 'node':
        return node_level_test(model, dataset, loader, args, device)
    elif level == 'link':
        return link_level_test(model, dataset, loader, args, device)
    elif level == 'graph':
        return graph_level_test(model, dataset, loader, args, device)


def main(args):
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
        if args.backbone == 'gcn_node':
            model = SugbCon(
                hidden_channels=args.num_hidden, encoder=Encoder(feature_dim, args.num_hidden),
                pool=Pool(in_channels=args.num_hidden),
                scorer=Scorer(args.num_hidden)).to(device)
        elif args.backbone == 'gcn_graph':
            model = SugbCon(
                hidden_channels=args.num_hidden, encoder=GNN_node(args.num_layers, args.num_hidden, args.in_drop, residual=args.residual, gnn_type=args.encoder, embed_mode=args.embed_mode),pool=Pool(in_channels=args.num_hidden),
                scorer=Scorer(args.num_hidden)).to(device)
        ## subgcon only supports subgraph
        args.split_mode = 'subgraph'
    elif args.method == 'graphmae':
        model = build_model(args).to(device)
    elif args.method == 'cotrain':
        model = build_model(args).to(device)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if use_scheduler:
        if args.lrtype == 'lambda':
            logging.info("Use scheduler lambdalr")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epochs) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            if args.warmup:
                warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
            else:
                warmup_scheduler = None
        elif args.lrtype == 'cosine':
            logging.info("Use scheduler cosine")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            if args.warmup:
                warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
            else:
                warmup_scheduler = None
    else:
        scheduler = None
        warmup_scheduler = None
    
    ## load datas
    datas = unify_dataset_loader(args.pre_train_datasets, args)
    ## saved stats
    best_avg_val = 0
    val_accs = []
    test_accs = []
    

    for e in range(max_epochs):
        ## pre-train stage
        for i, G in enumerate(datas):
            train_loader = G.search(batch_size = batch_size, shuffle = True, time='train')
            loss = train(model, optimizer, train_loader, args, scheduler, device, model_type = args.method, warmup_scheduler=warmup_scheduler)
            logging.info("Epoch {} Dataset {} Loss {}".format(e, args.pre_train_datasets[i], loss))
        ## evaluation part
        eval_val = []
        eval_test = []
        for i, G in enumerate(datas):
            test_loader = G.search(batch_size = batch_size, shuffle = False, time='test')

            val_res, test_res = test(model, G, test_loader, args, device, level = DATASET[args.pre_train_datasets[i]]['level'])
            eval_val.append(val_res)
            eval_test.append(test_res)
        for i in range(len(datas)):
            logging.info("Epoch {} Dataset {} Val-{} {} Test-{} {}".format(e, args.downstream_datasets[i], DATASET[args.downstream_datasets[i]]['metric'], eval_val[i], DATASET[args.downstream_datasets[i]]['metric'], eval_test[i]))
        avg_val = np.mean(eval_val)
        if avg_val > best_avg_val:
            best_avg_val = avg_val
            val_accs = eval_val
            test_accs = eval_test
    logging.info("Best Avg Val {}".format(best_avg_val))
    for i in range(len(datas)):
        logging.info("Dataset: {} Best Val {} Test {}".format(args.pre_train_datasets[i], val_accs[i], test_accs[i]))
    return best_avg_val, val_accs, test_accs    
            





    

if __name__ == '__main__':
    args = build_args()
    print(args)
    if not args.not_same_pretrain_downstream:
        args.downstream_datasets = args.pre_train_datasets
    main(args)
    

  
        
        
    

    
