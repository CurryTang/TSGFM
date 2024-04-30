import os.path as osp
from data.chemmol.gen_data import MolOFADataset
from graphmae.config import DATASET
from subgcon.efficient_dataset import NodeSubgraphDataset, LinkSubgraphDataset, GraphSubgraphDataset
import torch
import ogb 
from torch_geometric.utils import index_to_mask, to_undirected


def eval_task(metric):
    if metric == 'accuracy':
        evaluator = ogb.nodeproppred.Evaluator(name='ogbn-arxiv')
    elif metric == 'hits@100':
        evaluator = ogb.linkproppred.Evaluator(name='ogbl-collab')
    elif metric == 'f1':
        evaluator = ogb.graphproppred.Evaluator(name='ogbg-code2')
    elif metric == 'apr':
        evaluator = ogb.graphproppred.Evaluator(name='ogbg-molpcba')
    elif metric == 'auc':
        evaluator = ogb.graphproppred.Evaluator(name='ogbg-molhiv')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return evaluator


def load_one_tag_dataset(dataset = "cora", tag_data_path=""):
    AVAILABLE_DATASETS = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookhis', 
                          'bookchild', 'elephoto', 'elecomp', 'sportsfit', 'products', 'wikics', 
                          'cora-link', 'citeseer-link', 'pubmed-link', 'arxiv23-link', 'wikics-link']
    if dataset.endswith("-link"):
        dataset = dataset[:-5]
        link = True
    else:
        link = False
    lowers = [x.lower() for x in AVAILABLE_DATASETS]
    if dataset.lower() not in lowers:
        raise ValueError(f"Unknow dataset: {dataset}.")
    if tag_data_path == "":
        raise ValueError("tag_data_path is empty.")
    path = osp.join(tag_data_path, f"{dataset}/processed", "geometric_data_processed.pt")
    meta_data = osp.join(tag_data_path, f"{dataset}/processed", "data.pt")
    if not link:
        meta_class_info = meta_data['e2e_node']['class_node_text_feat'][1]
    else:
        meta_class_info = meta_data['e2e_link']['class_edge_text_feat'][1]
    if not osp.exists(path):
        raise ValueError(f"File not found: {path}")
    data = torch.load(path)[0]
    meta_class_emb = data.class_node_text_feat[meta_class_info]
    feature = data.node_text_feat
    data.y = data.y.view(-1)
    # edge_index = data.edge_index
    # if dataset != 'arxiv23':
    data.edge_index = to_undirected(data.edge_index)
    data.x = feature
    data.meta_class_emb = meta_class_emb
    m_size = data.x.size(0)
    ## the following is for downstream tasks
    if not link:
        if hasattr(data, "train_mask"):
            if isinstance(data.train_mask, list):
                train_mask = data.train_mask[0]
                val_mask = data.val_mask[0]
                test_mask = data.test_mask[0]
            else:
                train_mask = data.train_mask
                val_mask = data.val_mask
                test_mask = data.test_mask
        elif hasattr(data, 'train_masks'):
            train_mask = data.train_masks[0]
            val_mask = data.val_masks[0]
            test_mask = data.test_masks[0]
        elif hasattr(data, 'splits'):
            train_mask, val_mask, test_mask = [index_to_mask(data.splits['train'], size=m_size)], [index_to_mask(data.splits['valid'], size=m_size)], [index_to_mask(data.splits['test'], size = m_size)]
    if not link:
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    return data







def unify_dataset_loader(dataset_names, args):
    ds = []
    for d in dataset_names:
        level = DATASET[d]['level']
        task_dim = DATASET[d]['task_dim']
        if level == 'node':
            data = load_one_tag_dataset(d, args.tag_data_path)
            output_path = osp.join(args.cache_data_path, d)
            dataset = NodeSubgraphDataset(data, output_path, args.subgraph_size, name=d, sample=args.sample, split_mode=args.split_mode)
            ds.append(dataset)
        elif level == 'link':
            data = load_one_tag_dataset(d, args.tag_data_path)
            clear_d = d.split('-')[0]
            output_path = osp.join(args.cache_data_path, clear_d)
            dataset = LinkSubgraphDataset(data, output_path, args.subgraph_size, name=d, split_mode=args.split_mode)
            ds.append(dataset)
        elif level == 'graph':
            dataset = GraphSubgraphDataset(d, args.sb, args.tag_data_path, task_dim)
            ds.append(dataset)
        else:
            raise ValueError(f"Unknown dataset level: {level}")
    return ds