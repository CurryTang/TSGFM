#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200 
import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from subgcon.utils_mp import Subgraph, preprocess
from subgcon.models import SugbCon, Encoder, Scorer, Pool
from graphmae.utils import build_args, create_optimizer, set_random_seed, load_best_configs
from graphmae.data_util import load_downstream_dataset_pyg
import logging
import os.path as osp
from tqdm import tqdm

def train(model, optimizer, data, args, subgraph, scheduler, device):
    # Model training
    model.train()
    optimizer.zero_grad()
    sample_idx = random.sample(range(data.x.size(0)), args.batch_size)
    batch, index = subgraph.search(sample_idx)
    z, summary = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device))
    
    loss = model.loss(z, summary)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()

def get_all_node_emb(model, mask, num_node, subgraph, args, device):
    # Obtain central node embs from subgraphs 
    if isinstance(mask, list):
        mask = mask[0]
    node_list = np.arange(0, num_node, 1)[mask]
    list_size = node_list.size
    z = torch.Tensor(list_size, args.num_hidden).to(device)
    group_nb = math.ceil(list_size/args.batch_size)
    for i in tqdm(range(group_nb)):
        maxx = min(list_size, (i + 1) * args.batch_size)
        minn = i * args.batch_size 
        batch, index = subgraph.search(node_list[minn:maxx])
        node, _ = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), index.to(device))
        z[minn:maxx] = node
    return z
    
    
def test(model, data, subgraph, args, device):
    # Model testing
    model.eval()
    num_node = data.x.size(0)
    with torch.no_grad():
        train_z = get_all_node_emb(model, data.train_mask, num_node, subgraph, args, device)
        val_z = get_all_node_emb(model, data.val_mask, num_node, subgraph, args, device)
        test_z = get_all_node_emb(model, data.test_mask, num_node, subgraph, args, device)
    
    train_y = data.y[data.train_mask]
    val_y = data.y[data.val_mask]
    test_y = data.y[data.test_mask]
    val_acc, test_acc = model.test(train_z, train_y, val_z, val_y, test_z, test_y)
    print('val_acc = {} test_acc = {}'.format(val_acc, test_acc))
    return val_acc, test_acc



def main(args):
    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    graphs = load_downstream_dataset_pyg(args)
    feature_dim = graphs[0].x.size(1)
    ## pretrain part
    max_epochs = args.max_epoch
    use_scheduler = args.scheduler
    model = SugbCon(
        hidden_channels=args.num_hidden, encoder=Encoder(feature_dim, args.num_hidden),
        pool=Pool(in_channels=args.num_hidden),
        scorer=Scorer(args.num_hidden)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if use_scheduler:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epochs) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    
    ## saved stats
    best_avg_val = 0
    val_accs = []
    test_accs = []
    

    for e in range(max_epochs):
        ## pre-train stage
        for i, G in enumerate(graphs):
            ppr_path = osp.join('./subgraph/' + args.pre_train_datasets[i])
            subgraph = Subgraph(G.x, G.edge_index, ppr_path, args.subgraph_size, args.n_order, args.pre_train_datasets[i])
            subgraph.build()
            this_dataset_loss = train(model, optimizer, G, args, subgraph, scheduler, device)
            logging.info("Epoch {} Dataset {} Pretrain-Loss {}".format(e, i, this_dataset_loss))
        
        ## evaluation part
        eval_val = []
        eval_test = []
        for i, G in enumerate(graphs):
            ppr_path = osp.join('./subgraph/' + args.pre_train_datasets[i])
            subgraph = Subgraph(G.x, G.edge_index, ppr_path, args.subgraph_size, args.n_order, args.pre_train_datasets[i])
            subgraph.build()
            val_acc, test_acc = test(model, G, subgraph, args, device)
            eval_val.append(val_acc)
            eval_test.append(test_acc)
        for i in range(len(graphs)):
            logging.info("Epoch {} Dataset {} Val-Acc {} Test-Acc {}".format(e, args.pre_train_datasets[i], eval_val[i], eval_test[i]))
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
    if args.use_cfg != "":
        args = load_best_configs(args, args.use_cfg)
    print(args)
    main(args)
    

  
        
        
    

    
