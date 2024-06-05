#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200 
import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool
# from subgcon.models import SugbCon, Scorer, Pool
from graphmae.utils import build_args, create_optimizer, set_random_seed, load_best_configs
from graphmae.data_util import unify_dataset_loader
from graphmae.models import build_model, TaskHeadModel, build_model_backbone
from graphmae.evaluation import linear_test, link_linear_test, linear_mini_batch_test, zero_shot_eval
from graphmae.models.gcn import GNN_node
from graphmae.models.gcnvn import GNN_node_Virtualnode as GCNVN
from graphmae.models import build_model
from subgcon.ssl import DGIEncoder, dgi_train_step, BGRLEncoder, bgrl_train_step
from GCL.models import BootstrapContrast
import GCL.augmentors as A
import logging
from graphmae.config import DATASET
from ogb.linkproppred.evaluate import Evaluator
import pytorch_warmup as warmup
from GCL.models import SingleBranchContrast
import GCL.losses as L
from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt

def train(model, optimizer, loader, args, scheduler, device, model_type='subgcon', mask = None, head_name = None, warmup_scheduler = None, contrast_model = None):
    # Model training
    model.train()
    model = model.to(device)
    optimizer.zero_grad()
    avg_loss = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    for batch in loader:
        if batch.x.dim() == 1 and args.embed_mode == 'llm':
            batch.x = batch.node_text_feat
            batch.xe = batch.edge_text_feat
        if model_type == 'dgi':
            batch = batch.to(device)
            loss = dgi_train_step(model, contrast_model, batch, optimizer) 
        elif model_type == 'gcc':
            pass 
        elif model_type == 'bgrl':
            batch = batch.to(device)
            loss = bgrl_train_step(model, contrast_model, batch, optimizer)
        elif model_type == 'graphmae':
            if not hasattr(batch, 'batch') or not hasattr(batch, 'xe'):
                loss, loss_item = model(batch.x.to(device), batch.edge_index.to(device))
            else:
                loss, loss_item = model(batch.x.to(device), batch.edge_index.to(device), batch.xe.to(device), batch.batch.to(device))
        elif model_type == 'cotrain':
            ## mask and head_name should be provided for this case
            output = model(batch.x.to(device), batch.edge_index.to(device), head_name)
            loss = ce_loss(output[mask], batch.y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if model_type == 'bgrl':
            model.update_target_encoder(0.99)
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
    model = model.to(device)
    z = []
    size = len(loader)
    for batch in loader:
        if batch.x.dim() == 1 and args.embed_mode == 'llm':
            batch.x = batch.node_text_feat
            batch.xe = batch.edge_text_feat
        index = batch.ptr[:-1]
        if modeln == 'dgi':
            node, graph, _ = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
            if size > 1:
                node = node[index]
            else:
                node = node
        elif modeln == 'bgrl':
            h1, h2, _, _, _, _ = model(batch.x.to(device), batch.edge_index.to(device))
            out = torch.cat([h1, h2], dim=1)
            if size > 1:
                node = out[index]
            else:
                node = out
            graph = global_mean_pool(out, batch.batch.to(device))
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


def zero_shot_test(model, dataset, loader, args, device, level):
    model.eval()
    with torch.no_grad():
        z = get_all_node_emb(model, 1, loader, args, device, level = level, modeln=args.method)
    if level == 'link':
        evaluator = Evaluator(name='ogbl-ppa')
    else:
        evaluator = None
    dataset.data.num_classes = dataset.data.y.max().item() + 1
    val_acc, test_acc = zero_shot_eval(z, dataset, device, evaluator, mode=level, m_name=DATASET[dataset.name]['metric'])
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
    logging.info(args)
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
    if args.method == 'gppt':
        model = Edgepred_GPPT(dataset_name = args.pre_train_datasets, gnn_type = args.encoder, hid_dim = args.num_hidden, gln = args.num_layers, num_epoch=args.epochs, device=args.device)
    elif args.method == 'gprompt':
        model = Edgepred_Gprompt(dataset_name = args.pre_train_datasets, gnn_type = args.encoder, hid_dim = args.num_hidden, gln = args.num_layers, num_epoch=args.epochs, device=args.device)
    elif args.method == 'graphmae':
        model = build_model(args).to(device)
        contrast_model = None
    elif args.method == 'dgi':
        model = build_model_backbone(args, feature_dim, args.num_hidden).to(device)
        model = DGIEncoder(model, args.num_hidden).to(device)
        contrast_model = SingleBranchContrast(
            loss=L.JSD(), mode='G2L'
        ).to(device)
    elif args.method == 'bgrl':
        model = build_model_backbone(args, feature_dim, args.num_hidden).to(device)
        aug1 = A.Compose([A.EdgeRemoving(pe=args.pe), A.FeatureMasking(pf=args.pf)])
        aug2 = A.Compose([A.EdgeRemoving(pe=args.pe), A.FeatureMasking(pf=args.pf)])
        model = BGRLEncoder(model, augmentor=(aug1, aug2), hidden_dim=args.num_hidden, dropout=args.in_drop, predictor_norm=args.norm).to(device)
        contrast_model = BootstrapContrast(
            loss=L.BootstrapLatent(), mode='L2L').to(device)
    elif args.method == 'gcc':
        model = build_model_backbone(args, feature_dim, args.num_hidden).to(device)
    elif args.method == 'cotrain':
        model = build_model(args).to(device)
        contrast_model = None
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        model = model.to(device)
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
    
    saved_model_name = f"{args.method}-{args.encoder}-{args.decoder}-{args.pre_train_datasets}.pt"

    if args.eval_only:
        max_epochs = 1

    for e in range(max_epochs):
        if not args.eval_only:
        ## pre-train stage
            for i, G in enumerate(datas):
                train_loader = G.search(batch_size = batch_size, shuffle = True, time='train', infmode='cpu' if args.cpuinf else 'gpu')
                loss = train(model, optimizer, train_loader, args, scheduler, device, model_type = args.method, warmup_scheduler=warmup_scheduler, contrast_model = contrast_model)
                logging.info("Epoch {} Dataset {} Loss {}".format(e, args.pre_train_datasets[i], loss))
        ## evaluation part
        eval_val = []
        eval_test = []
        for i, G in enumerate(datas):
            test_loader = G.search(batch_size = batch_size, shuffle = False, time='test', infmode='cpu' if args.cpuinf else 'gpu')

            this_level = DATASET[args.pre_train_datasets[i]]['level']
            val_res, test_res = test(model, G, test_loader, args, 'cpu' if args.cpuinf else 'cuda', level = this_level)
            eval_val.append(val_res)
            eval_test.append(test_res)
        for i in range(len(datas)):
            logging.info("Epoch {} Dataset {} Val-{} {} Test-{} {}".format(e, args.downstream_datasets[i], DATASET[args.downstream_datasets[i]]['metric'], eval_val[i], DATASET[args.downstream_datasets[i]]['metric'], eval_test[i]))
        avg_val = np.mean(eval_val)
        if avg_val > best_avg_val:
            best_avg_val = avg_val
            val_accs = eval_val
            test_accs = eval_test
            if args.save_model:
                torch.save(model.state_dict(), saved_model_name)
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
    

  
        
        
    

    
