from torch_geometric.data import InMemoryDataset
from abc import ABC, abstractmethod
import torch_geometric as pyg
from torch.utils.data import Dataset
from tqdm import trange
from subgcon.efficient_ppr import topk_ppr_matrix
from graphllm.utils import get_mask
from data.chemmol.gen_data import MolOFADataset
import sys 
sys.path.append("..")
from utils import SentenceEncoder
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler
from torch_geometric.transforms.random_link_split import RandomLinkSplit
import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.utils import index_to_mask
from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from tqdm import tqdm


def keep_attrs_for_data(data):
    for k in data.keys():
        if k not in ['x', 'edge_index', 'edge_attr', 'y']:
            delattr(data, k)
    return data


class SubgraphDataset(InMemoryDataset, ABC):
    @abstractmethod
    def build(self):
        """
            Generate and cache the subgraphs
        """
    
    @abstractmethod
    def search(self, data_list, batch_size, shuffle):
        """
            For generating the embedding 
            Given a list: data_list, 
            this can be a list of node indexes, a list of edge indexes, or a list of graph indices
        """



class NodeSubgraphDataset(SubgraphDataset):
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora', sample = 1, split_mode = 'subgraph'):
        ## Split can be either subgraph or graphsaint
        super(NodeSubgraphDataset, self).__init__()
        self.link = 'node'
        self.name = name
        self.output_path = output_path
        self.output_name = osp.join(output_path, f"{name}_subgraphs_nc_neighbors_{topk}.pt")
        self.data = data
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        train_mask, val_mask, test_mask = get_mask(data)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask
        self.csr_data = pyg.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()
        self.num_nodes = self.data.x.size(0)
        self.sample = sample
        self.split_mode = split_mode
        self.saint_loader = None
        self.train_loader = None
        self.test_loader = None
        self.build() 

    def build(self):
        if self.topk == -1:
            return
        if self.split_mode == 'graphsaint':
            os.makedirs(self.output_path, exist_ok=True)
            self.saint_loader = GraphSAINTRandomWalkSampler(self.data, batch_size=self.topk, walk_length=2, num_steps=5, sample_coverage=100, save_dir=self.output_path)
            return
        print(f"Building subgraphs for node-level dataset {self.name}")
        idx = torch.arange(self.num_nodes)
        ## here's a tradeoff between randomness and efficiency
        self.data = keep_attrs_for_data(self.data)
        train_loader = ShaDowKHopSampler(self.data, depth=2, num_neighbors=10, node_idx=idx, batch_size=128, num_workers=12, shuffle=True)
        self.train_loader = train_loader 
        test_loader = ShaDowKHopSampler(self.data, depth=2, num_neighbors=10, node_idx=idx, batch_size=128, num_workers=12, shuffle=False)
        self.test_loader = test_loader

    def search(self, batch_size = 32, shuffle = False, time='train', infmode='cpu'):
        #Extract subgraphs for nodes in the list
        if self.topk != -1:
            if self.split_mode == 'graphsaint':
                if time == 'train':
                    return self.saint_loader
                else:
                    if infmode == 'cpu':
                        return DataLoader([self.data], batch_size=1, shuffle=False)
                    else:
                        if self.test_loader is not None:
                            return self.test_loader
                        else:
                            self.build()
                            return self.test_loader
            else:
                if self.train_loader is None:
                    self.build()
                return self.train_loader if time == 'train' else self.test_loader
        else:
            return DataLoader([self.data], batch_size=1, shuffle=False)
    

class LinkSubgraphDataset(SubgraphDataset):
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora', train_ratio=.7, val_ratio=.1, sample=1, split_mode='subgraph'):
        super().__init__()
        self.level = 'link'
        self.name = name
        self.output_path = output_path
        self.output_name = osp.join(output_path, f"{self.name}_subgraphs_nc_neighbors_{topk}.pt")
        self.data = data
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        splits = RandomLinkSplit(num_val=val_ratio, num_test=1-train_ratio-val_ratio, is_undirected=True, add_negative_train_samples=False)
        train_data, val_data, test_data = splits(data)
        self.edge_index = train_data.edge_index
        self.train_edge_index = train_data.edge_label_index
        self.data.edge_index = train_data.edge_index
        self.data.train_edge_index = train_data.edge_label_index
        val_edge_index = val_data.edge_label_index
        test_edge_index = test_data.edge_label_index
        train_edge_label = train_data.edge_label
        val_edge_label = val_data.edge_label
        test_edge_label = test_data.edge_label
        self.pos_val_edge_index = val_edge_index[:, val_edge_label == 1]
        self.pos_test_edge_index = test_edge_index[:, test_edge_label == 1]
        self.neg_val_edge_index = val_edge_index[:, val_edge_label == 0]
        self.neg_test_edge_index = test_edge_index[:, test_edge_label == 0]
        self.csr_data = pyg.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()
        self.num_nodes = self.data.x.size(0)
        self.sample = sample
        self.split_mode = split_mode
        self.saint_loader = None
        self.train_loader = None
        self.test_loader = None
        self.build() 
    
    def build(self):
        if self.topk == -1:
            return
        if self.split_mode == 'graphsaint':
            os.makedirs(self.output_path, exist_ok=True)
            self.saint_loader = GraphSAINTRandomWalkSampler(self.data, batch_size=self.topk, walk_length=2, num_steps=5, sample_coverage=100, save_dir=self.output_path)
            return
        idx = torch.arange(self.num_nodes)
        ## here's a tradeoff between randomness and efficiency
        self.data = keep_attrs_for_data(self.data)
        train_loader = ShaDowKHopSampler(self.data, depth=2, num_neighbors=10, node_idx=idx, batch_size=128, num_workers=12, shuffle=True)
        self.train_loader = train_loader
        test_loader = ShaDowKHopSampler(self.data, depth=2, num_neighbors=10, node_idx=idx, batch_size=128, num_workers=12, shuffle=False)
        self.test_loader = test_loader

    def search(self, batch_size = 32, shuffle = False, time='train', infmode='cpu'):
        #Extract subgraphs for nodes in the list
        if self.topk != -1:
            if self.split_mode == 'graphsaint':
                if time == 'train':
                    return self.saint_loader
                else:
                    if infmode == 'cpu':
                        return DataLoader([self.data], batch_size=1, shuffle=False)
                    else:
                        if self.test_loader is not None:
                            return self.test_loader
                        else:
                            self.build()
                            return self.test_loader
            else:
                if self.train_loader is None:
                    self.build()
                return self.train_loader if time == 'train' else self.test_loader
        else:
            return DataLoader([self.data], batch_size=1, shuffle=False)

class GraphSubgraphDataset(SubgraphDataset):
    def __init__(self, dataset_name, sb_path, mol_cache_path, num_classes = 2, use_llm = True):
        super().__init__()
        self.level = 'graph'
        self.name = dataset_name
        encoder = SentenceEncoder("minilm", root=sb_path, batch_size=256)
        self.dataset = MolOFADataset(name = dataset_name, encoder=encoder, root=mol_cache_path, load_text=True)
        self.num_nodes = len(self.dataset)
        idxs = self.dataset.get_idx_split()
        train_idx = torch.tensor(idxs['train'])
        val_idx = torch.tensor(idxs['valid'])
        test_idx = torch.tensor(idxs['test'])
        self.train_mask = index_to_mask(train_idx, size=self.num_nodes)
        self.val_mask = index_to_mask(val_idx, size=self.num_nodes)
        self.test_mask = index_to_mask(test_idx, size=self.num_nodes)
        self.num_class = num_classes

        
    
    def build(self):
        pass 

    
    def search(self, batch_size = 32, shuffle = False, time='train'):
        return DataLoader(self.dataset, batch_size=32, shuffle=False)
         
    @property
    def num_classes(self):
        return self.num_class