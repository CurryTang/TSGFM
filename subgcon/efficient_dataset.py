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
from torch_geometric.loader import DataLoader
import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.utils import index_to_mask


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
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora'):
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
        self.csr_data = pyg.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()
        self.num_nodes = self.data.x.size(0)
        self.build() 

    def build(self):
        if osp.exists(self.output_name):
            print(f"Loading subgraphs for node-level dataset {self.name}")
            self.subgraphs = torch.load(self.output_name)
            return
        self.subgraphs = []
        print(f"Building subgraphs for node-level dataset {self.name}")
        idx = np.arange(self.num_nodes)
        ppr_matrix = topk_ppr_matrix(self.csr_data, self.alpha, self.eps, idx, self.topk)
        for i in trange(self.num_nodes):
            nodes = torch.tensor(ppr_matrix[i].indices)
            x = self.data.x[nodes]
            labels = self.data.y[i]
            subg, _ = pyg.utils.subgraph(nodes, self.data.edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
            self.subgraphs.append(pyg.data.Data(x=x, edge_index=subg, y=labels))
        dir_name = osp.dirname(self.output_name)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.subgraphs, self.output_name)

    def search(self, batch_size = 32, shuffle = False):
        #Extract subgraphs for nodes in the list
        return DataLoader(self.subgraphs, batch_size=batch_size, shuffle=shuffle)
    

class LinkSubgraphDataset(SubgraphDataset):
    def __init__(self, data, output_path, topk=50, alpha=0.85, eps=1e-9, name = 'cora', train_ratio=.7, val_ratio=.1):
        super().__init__()
        self.level = 'link'
        self.name = name
        self.output_path = output_path
        self.output_name = osp.join(output_path, f"{name}_subgraphs_lp_neighbors_{topk}.pt")
        self.data = data
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.train_ratio = train_ratio 
        self.val_ratio = val_ratio
        splits = self.cite_link_splitter(data)["train"]
        self.train_idx = splits["train"]
        self.val_idx = splits["valid"]
        self.test_idx = splits["test"]
        ## remove test link
        self.train_edge_index = data.edge_index[:, self.train_idx]
        self.csr_data = pyg.utils.to_scipy_sparse_matrix(self.train_edge_index).tocsr()
        self.num_nodes = self.data.x.size(0)
        self.build() 
    
    def cite_link_splitter(self, data):
        edges = data.edge_index
        edge_perm = torch.randperm(len(edges[0]))
        train_offset = int(len(edge_perm) * (self.train_ratio))
        val_offset = int(len(edge_perm) * (self.train_ratio + self.val_ratio))
        edge_indices = {"train": edge_perm[:train_offset], "valid": edge_perm[train_offset:val_offset],
                        "test": edge_perm[val_offset:], }
        return edge_indices

    def build(self):
        self.subgraphs = []
        print(f"Building subgraphs for link-level dataset {self.name}")
        # idx = torch.arange(self.num_links)
        ppr_matrix = topk_ppr_matrix(self.csr_data, self.alpha, self.eps, idx, self.topk)
        for i in trange(self.num_nodes):
            nodes = torch.tensor(ppr_matrix[i].indices)
            x = self.data.x[nodes]
            labels = self.data.y[i]
            subg, _ = pyg.utils.subgraph(nodes, self.train_edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
            self.subgraphs.append(pyg.data.Data(x=x, edge_index=subg))
        torch.save(self.subgraphs, self.output_name)
    
    def search(self, data_list, batch_size=32, shuffle=False):
        #Extract subgraphs for nodes in the list
        return DataLoader(self.subgraphs, batch_size=batch_size, shuffle=shuffle)

class GraphSubgraphDataset(SubgraphDataset):
    def __init__(self, dataset_name, sb_path, mol_cache_path):
        super().__init__()
        self.level = 'graph'
        encoder = SentenceEncoder("minilm", root=sb_path, batch_size=256)
        self.dataset = MolOFADataset(name = dataset_name, encoder=encoder, root=mol_cache_path, load_text=True)
        self.num_nodes = len(self.dataset)
        idxs = self.dataset.get_idx_split()
        train_idx = torch.tensor(idxs['train'])
        val_idx = torch.tensor(idxs['val'])
        test_idx = torch.tensor(idxs['test'])
        self.train_mask = index_to_mask(train_idx, size=self.num_nodes)
        self.val_mask = index_to_mask(val_idx, size=self.num_nodes)
        self.test_mask = index_to_mask(test_idx, size=self.num_nodes)
        
    
    def build(self):
        pass 

    
    def search(self, data_list):
        return DataLoader(self.dataset, batch_size=32, shuffle=False)
         
