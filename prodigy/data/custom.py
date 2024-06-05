import os
import random
from collections import defaultdict
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import sys
from .sampler import NeighborSamplerCacheAdj
from .dataset import SubgraphDataset
from .dataloader import NeighborTask, MultiTaskSplitWay, MultiTaskSplitBatch, MulticlassTask, ParamSampler, BatchSampler, Collator, ContrastiveTask
from .augment import get_aug


class CustomDataset(SubgraphDataset):
    def __init__(self, graph, neighbor_sampler, offset=0, bidirectional=True, node_graph = False):
        super().__init__(graph, neighbor_sampler, offset, bidirectional, node_graph)

        self.node_attrs = ['x']
        self.edge_attrs = []


def process_mask(data):
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
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data



def get_a_custom_dataset(root = '../MyOFA/cache_data_minilm', dataset_name = "cora", n_hop=2, **kwargs):
    # dataset = MAG240MDataset(root)
    pyg_data_obj = torch.load(os.path.join(root, dataset_name, 'processed', "geometric_data_processed.pt"))[0]
    pyg_data_obj = process_mask(pyg_data_obj)
    pyg_data_obj.x = pyg_data_obj.node_text_feat
    delattr(pyg_data_obj, "node_text_feat")
    neighbor_sampler = NeighborSamplerCacheAdj(os.path.join(root, f"{dataset_name}_adj_bi.pt"), pyg_data_obj, n_hop)
    print(f"Done loading {dataset_name} neighbor sampler.")
    return CustomDataset(pyg_data_obj, neighbor_sampler)



def custom_labels(split, dataset, node_split = "", root="dataset", train_cap=3):
    label = dataset.graph.y.view(-1).numpy().copy()
    label_set = range(dataset.graph.y.max().item() + 1)
    train_label = dataset.graph.y.view(-1).numpy().copy()
    split_idx = dataset.graph.train_mask.numpy()
    train_label[split_idx] = -1 - train_label[split_idx]
    train_label = -1 - train_label
    COUNT_CAP = train_cap
    if COUNT_CAP is not None:
        # This only matters if we finetuning
        for i in range(dataset.graph.y.max().item() + 1):
            idx = (train_label == i)
            if idx.sum() > COUNT_CAP:
                disabled_idx = np.where(idx)[0][COUNT_CAP:]
                train_label[disabled_idx] = -1 - i
    if split == "train": 
        label = dataset.graph.y.view(-1).numpy()
        train_label = None
    else:
        if split == 'val':
            split_idx = dataset.graph.val_mask.numpy()
        elif split == 'test':
            split_idx = dataset.graph.test_mask.numpy()
        # split_idx = dataset.get_idx_split()[split if split != "val" else "valid"].numpy()
        label[split_idx] = -1 - label[split_idx]
        label = -1 - label

    linear_probe = False
    return MulticlassTask(label, label_set, train_label, linear_probe), label, label_set

def get_custom_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, num_workers, aug, aug_test, train_cap, **kwargs):
    seed = sum(ord(c) for c in split)
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.graph.x)
    else:
        aug = get_aug("")
    task, labels, label_set = custom_labels(split, dataset, node_split, train_cap)
    neighbor_sampler = copy.copy(dataset.neighbor_sampler)
    neighbor_sampler.num_hops = 2
    
    if task_name.endswith("sb"):
        task_base = MultiTaskSplitBatch([
            task,
            NeighborTask(neighbor_sampler, len(dataset), "inout")
        ], ["mct", "nt"], [1, 3])
    elif task_name.endswith("sw"):
        task_base = MultiTaskSplitWay([
            task, 
            NeighborTask(neighbor_sampler, len(dataset), "inout")
        ], ["mct", "nt"], split="even")
    
    sampler = BatchSampler(
        batch_count,
        task_base,
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=seed,
    )
    label_meta = {}
    num_classes = len(label_set)
    label_meta["mct"] = torch.zeros(1, 768).expand(num_classes, -1)
    label_meta["nt"] = torch.zeros(1, 768).expand(len(dataset), -1)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_meta, aug=aug))
    return dataloader





if __name__ == "__main__":
    from tqdm import tqdm
    import cProfile

    root = "../FSdatasets/mag240m"
    n_hop = 2

    dataset = get_mag240m_dataset(root, n_hop)
    dataloader = get_mag240m_dataloader(dataset, "train", "", 5, 3, 3, 24, 10000, root, 10)

    test = next(iter(dataloader))
    import ipdb; ipdb.set_trace()
