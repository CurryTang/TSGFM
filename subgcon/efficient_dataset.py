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


class LabelPerClassSplit(object):
    """
    Class for splitting data into training, validation, and test sets based on labels.

    This class provides a callable object for splitting data into training, validation, and test sets.
    The splitting is done based on the labels of the data, with a specified number of labels per class for the training set.
    """
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
            num_test: int = 1000,
            inside_old_mask: bool = False
    ):
        """
        Constructor method for the LabelPerClassSplit class.

        Initializes a new instance of the LabelPerClassSplit class with the provided parameters.

        Parameters:
        num_labels_per_class (int, optional): The number of labels per class for the training set. Defaults to 20.
        num_valid (int, optional): The number of validation data points. Defaults to 500.
        num_test (int, optional): The number of test data points. If -1, all remaining data points after training and validation are used for testing. Defaults to -1.
        inside_old_mask (bool, optional): Whether to consider only data points inside the old mask for splitting. Defaults to False.

        Returns:
        None
        """
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid
        self.num_test = num_test
        self.inside_old_mask = inside_old_mask

    def __call__(self, y, total_num):
        """
        Callable method for the LabelPerClassSplit class.

        This method splits the data into training, validation, and test sets based on the labels of the data.

        Parameters:
        data: The data to be split.
        total_num (int): The total number of data points.

        Returns:
        tuple: A tuple containing the masks for the training, validation, and test sets.
        """
        new_train_mask = torch.zeros(total_num, dtype=torch.bool)
        new_val_mask = torch.zeros(total_num, dtype=torch.bool)
        new_test_mask = torch.zeros(total_num, dtype=torch.bool)

        perm = torch.randperm(total_num)
        train_cnt = np.zeros(y.max().item() + 1, dtype=np.int32)

        for i in range(perm.numel()):
            label = y[perm[i]]
            if train_cnt[label] < self.num_labels_per_class:
                train_cnt[label] += 1
                new_train_mask[perm[i]] = 1
            elif new_val_mask.sum() < self.num_valid:
                new_val_mask[perm[i]] = 1
            else:
                if new_test_mask.sum() < self.num_test:
                    new_test_mask[perm[i]] = 1
                else:
                    break
        
        if self.num_test == -1:
            new_test_mask = ~new_train_mask & ~new_val_mask

        return new_train_mask, new_val_mask, new_test_mask


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


def update_few_shot_train_mask(y, num_labels_per_class = 3):
    splitter = LabelPerClassSplit(num_labels_per_class=num_labels_per_class)
    num_nodes = y.size(0)
    y = y.to(torch.long)
    train_mask, _, _ = splitter(y, num_nodes)
    return train_mask


class NodeSubgraphDataset(SubgraphDataset):
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora', sample = 1, split_mode = 'subgraph', few_shot = False):
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
        if few_shot:
            self.data.train_mask = update_few_shot_train_mask(data.y)
            self.train_mask = self.data.train_mask
        self.build() 

    def build(self, time='train'):
        if self.topk == -1:
            return
        if self.split_mode == 'graphsaint' and time == 'train':
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
                            self.build(time='test')
                            return self.test_loader
            else:
                if self.train_loader is None:
                    self.build()
                return self.train_loader if time == 'train' else self.test_loader
        else:
            return DataLoader([self.data], batch_size=1, shuffle=False)
    

class LinkSubgraphDataset(SubgraphDataset):
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora', train_ratio=.7, val_ratio=.1, sample=1, split_mode='subgraph', few_shot=False):
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
    
    def build(self, time='train'):
        if self.topk == -1:
            return
        if self.split_mode == 'graphsaint' and time == 'train':
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
                            self.build(time='test')
                            return self.test_loader
            else:
                if self.train_loader is None:
                    self.build()
                return self.train_loader if time == 'train' else self.test_loader
        else:
            return DataLoader([self.data], batch_size=1, shuffle=False)

class GraphSubgraphDataset(SubgraphDataset):
    def __init__(self, dataset_name, sb_path, mol_cache_path, num_classes = 2, use_llm = True, few_shot = False):
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
        if few_shot:
            self.train_mask = update_few_shot_train_mask(self.dataset.y)
        
    
    def build(self):
        pass 

    
    def search(self, batch_size = 32, shuffle = False, time='train', infmode='cpu'):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
         
    @property
    def num_classes(self):
        return self.num_class