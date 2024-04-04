import os
import numpy as np
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from graphmae.utils import mask_edge
import logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
from dgl.dataloading import SAINTSampler, DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label

def drop_edge(g, drop_edge_rate):
    if drop_edge_rate <= 0:
        return g, g

    g = g.remove_self_loop()
    mask_index1 = mask_edge(g, drop_edge_rate)
    mask_index2 = mask_edge(g, drop_edge_rate)
    g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
    g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
    return g1, g2


class OnlineLCLoader(DataLoader):
    def __init__(self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["x"] = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels != None:
            label = self.labels[batch_idx]
        else:
            label = None
        
        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes
        

def setup_training_dataloder(loader_type, training_nodes, graph, feats, batch_size, drop_edge_rate=0, pretrain_clustergcn=False, cluster_iter_data=None):
    num_workers = 8

    if loader_type == "lc":
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")
    
    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(training_nodes, graph, feats=feats, drop_edge_rate=drop_edge_rate, batch_size=batch_size, shuffle=True, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader

def setup_multiple_training_dataloder(loader_type, training_nodes, graphs, feats, batch_size, drop_edge_rate=0, pretrain_clustergcn=False, cluster_iter_data=None):
    multiple_loaders = []

    for G in graphs:
        dataloader = setup_training_dataloder(loader_type, training_nodes, G, feats, batch_size, drop_edge_rate, pretrain_clustergcn, cluster_iter_data)
        multiple_loaders.append(dataloader)
    
    return multiple_loaders


def setup_saint_dataloader(args, graphs):
    datasets = args.pre_train_datasets
    loaders = []
    sampler = SAINTSampler(mode='node', budget=args.sg_size)
    for i, G_dict in enumerate(graphs):
        G = G_dict['g']
        base_iter = args.num_iters 
        graph_size = G.ndata['x'].shape[0]
        cal_iter = max(int(graph_size / args.sg_size), 1)
        num_iters = base_iter * cal_iter
        ## reptition considered later
        r = G_dict['r']
        n = G_dict['name']
        subgraph_loader = DataLoader(G, torch.arange(num_iters), sampler, num_workers=4)
        loaders.append(subgraph_loader)
    return loaders



def setup_eval_dataloder(loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False):
    num_workers = 8
    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, persistent_workers=True, num_workers=num_workers)
    return dataloader


def setup_finetune_dataloder(loader_type, graph, feats, ego_graph_nodes, labels, batch_size, shuffle=False):
    num_workers = 8

    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")
    
    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, labels=labels, feats=feats, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, persistent_workers=True)
    
    return dataloader