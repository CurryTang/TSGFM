import os.path as osp
from data.chemmol.gen_data import MolOFADataset
from graphmae.config import DATASET
from subgcon.efficient_dataset import NodeSubgraphDataset, LinkSubgraphDataset, GraphSubgraphDataset
import torch

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
    if not osp.exists(path):
        raise ValueError(f"File not found: {path}")
    data = torch.load(path)[0]
    feature = data.node_text_feat
    edge_index = data.edge_index
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
    data.x = feature
    if not link:
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    return data







def unify_dataset_loader(dataset_names, args):
    ds = []
    for d in dataset_names:
        level = DATASET[d]['level']
        if level == 'node':
            data = load_one_tag_dataset(d, args.tag_data_path)
            output_path = osp.join(args.cache_data_path, d)
            dataset = NodeSubgraphDataset(data, output_path, args.subgraph_size, name=d)
            ds.append(dataset)
        elif level == 'link':
            data = load_one_tag_dataset(d, args.tag_data_path)
            output_path = osp.join(args.cache_data_path, d)
            dataset = LinkSubgraphDataset(data, output_path, args.subgraph_size, name=d)
            ds.append(dataset)
        elif level == 'graph':
            dataset = GraphSubgraphDataset(d, args.sb, args.subgraph_size)
            ds.append(dataset)
        else:
            raise ValueError(f"Unknown dataset level: {level}")
    return ds