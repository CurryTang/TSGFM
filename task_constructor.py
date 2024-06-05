import torch
import torch_geometric as pyg
import json
import numpy as np
import copy
import random
import utils
from data.KG.gen_data import KGOFADataset
from data.chemmol.gen_data import MolOFADataset
from data.single_graph.gen_data import SingleGraphOFADataset
from data.mag240m.gen_data import SegmentDataset
import os

from ofa_datasets import (GraphListDataset, SubgraphDataset, MultiDataset, GraphListHierDataset, SubgraphHierDataset,
                          SubgraphLinkHierDataset, SubgraphKGHierDataset, SubgraphNopromptDataset,
                          GraphListNopromptDataset, SubgraphNopromptLinkDataset, FewShotDataset, MassiveDataset)
from fs_datamanager import FewShotDataManager, SimpleFSManager, LowRateLabelManager

from gp.utils.utils import k_fold_ind, k_fold2_split
from gp.lightning.data_template import DataWithMeta

# TODO: Instead of using global() to access these functions, come up with something more elegant
from gp.lightning.metric import (binary_auc_func, flat_binary_func, classification_func, EvalKit, )
from utils import (binary_apr_func, binary_auc_multi_func, binary_single_auc_func, classification_single_func,
                   flat_auc, )

from ogb.nodeproppred import PygNodePropPredDataset

name2dataset = {"arxiv": SingleGraphOFADataset, "arxivyear": SingleGraphOFADataset, "cora": SingleGraphOFADataset, "pubmed": SingleGraphOFADataset,
                'citeseer': SingleGraphOFADataset, 'arxiv23': SingleGraphOFADataset, "WN18RR": KGOFADataset, "FB15K237": KGOFADataset, "wikics": SingleGraphOFADataset, "bookchild": SingleGraphOFADataset, "amazonratings": SingleGraphOFADataset, "bookhis": SingleGraphOFADataset, "elecomp": SingleGraphOFADataset, "elephoto": SingleGraphOFADataset, "sportsfit": SingleGraphOFADataset, 'products': SingleGraphOFADataset,
                "chemblpre": MolOFADataset, "chempcba": MolOFADataset, "chemhiv": MolOFADataset, "bace": MolOFADataset, "bbbp": MolOFADataset, 
                "muv": MolOFADataset, "toxcast": MolOFADataset, "tox21": MolOFADataset, 'mag240m': SegmentDataset, 'dblp': SingleGraphOFADataset, 'tolokers': SingleGraphOFADataset}


saved_edge_index = {}


########################################################################
# Dataset split functions, split datasets into train/valid/test splits #
########################################################################

def generate_train_val_test_masks(dataset_size, train_ratio, validation_ratio, test_ratio):
    """Generates training, validation, and testing masks as PyTorch tensors.

    Args:
        dataset_size: The total number of data points in the dataset.
        train_ratio: The proportion of data to be used for training.
        validation_ratio: The proportion of data to be used for validation.
        test_ratio: The proportion of data to be used for testing.

    Returns:
        tuple: A tuple containing the training mask, validation mask, and testing mask.
    """

    if train_ratio + validation_ratio + test_ratio != 1:
        raise ValueError("Ratios must sum up to 1")

    num_train = int(dataset_size * train_ratio)
    num_val = int(dataset_size * validation_ratio)
    num_test = dataset_size - num_train - num_val

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_mask = torch.zeros(dataset_size, dtype=torch.bool)
    train_mask[indices[:num_train]] = True

    val_mask = torch.zeros(dataset_size, dtype=torch.bool)
    val_mask[indices[num_train:num_train + num_val]] = True

    test_mask = torch.zeros(dataset_size, dtype=torch.bool)
    test_mask[indices[num_train + num_val:]] = True

    return train_mask, val_mask, test_mask

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

    def __call__(self, data, total_num):
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
        train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int32)

        for i in range(perm.numel()):
            label = data.y[perm[i]]
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

def ArxivSplitter(dataset):
    text_g = dataset.data
    kfold = k_fold_ind(text_g.y, 10)
    text_split = k_fold2_split(kfold, len(text_g.y))[0]
    split = {}
    split["train"] = text_split[0]
    split["valid"] = text_split[1]
    split["test"] = text_split[2]
    return split

def OGB_Splitter(dataset):
    if dataset.name == 'arxiv' or dataset.name == 'arxivyear':
        ogb_data = PygNodePropPredDataset(name='ogbn-arxiv', root=dataset.data_dir)
        ogb_splits = ogb_data.get_idx_split()
        split = {}
        split["train"] = ogb_splits['train']
        split["valid"] = ogb_splits['valid']
        split["test"] = ogb_splits['test']
    elif dataset.name == 'arxiv23':
        text_g = dataset.data
        split = {}
        split["train"] = torch.from_numpy(text_g.train_id).long()
        split["valid"] = torch.from_numpy(text_g.val_id).long()
        split["test"] = torch.from_numpy(text_g.test_id).long()
    return split


def ArxivFSSplitter(dataset):
    labels = dataset.data.y
    with open("data/low_resource_split.json", "r") as f:
        lr_class_split = json.load(f)
    arxiv_cls_split = lr_class_split["arxiv"]
    fs_split = []
    for split in arxiv_cls_split:
        cls_idx = []
        data_idx = []
        for cls in split:
            cls_idx.append(cls)
            cls_data_idx = (labels == cls).nonzero(as_tuple=True)[0]
            data_idx.append(cls_data_idx.numpy())
        fs_split.append([np.array(cls_idx), data_idx])
    return {"train": fs_split[0], "valid": fs_split[1], "test": fs_split[2]}


def ProductsFSSplitter(dataset):
    labels = dataset.data.y
    with open("data/low_resource_split.json", "r") as f:
        lr_class_split = json.load(f)
    arxiv_cls_split = lr_class_split["products"]
    fs_split = []
    for split in arxiv_cls_split:
        cls_idx = []
        data_idx = []
        for cls in split:
            cls_idx.append(cls)
            cls_data_idx = (labels == cls).nonzero(as_tuple=True)[0]
            data_idx.append(cls_data_idx.numpy())
        fs_split.append([np.array(cls_idx), data_idx])
    return {"train": fs_split[0], "valid": fs_split[1], "test": fs_split[2]}


def SportsfitFSSplitter(dataset):
    labels = dataset.data.y
    with open("data/low_resource_split.json", "r") as f:
        lr_class_split = json.load(f)
    arxiv_cls_split = lr_class_split["sportsfit"]
    fs_split = []
    for split in arxiv_cls_split:
        cls_idx = []
        data_idx = []
        for cls in split:
            cls_idx.append(cls)
            cls_data_idx = (labels == cls).nonzero(as_tuple=True)[0]
            data_idx.append(cls_data_idx.numpy())
        fs_split.append([np.array(cls_idx), data_idx])
    return {"train": fs_split[0], "valid": fs_split[1], "test": fs_split[2]}



def CiteSplitter(dataset):
    text_g = dataset.data
    split = {"train": text_g.train_masks[0].nonzero(as_tuple=True)[0],
             "valid": text_g.val_masks[0].nonzero(as_tuple=True)[0],
             "test": text_g.test_masks[0].nonzero(as_tuple=True)[0], }
    return split

def OfficialSplitter(dataset):
    text_g = dataset.data
    split = {"train": text_g.train_mask.nonzero(as_tuple=True)[0],
             "valid": text_g.val_mask.nonzero(as_tuple=True)[0],
             "test": text_g.test_mask.nonzero(as_tuple=True)[0], }
    return split

def CiteHigh(dataset):
    text_g = dataset.data
    num_nodes = text_g.x.size(0)
    train_mask, val_mask, test_mask = generate_train_val_test_masks(num_nodes, 0.6, 0.2, 0.2)
    split = {"train": train_mask.nonzero(as_tuple=True)[0],
             "valid": val_mask.nonzero(as_tuple=True)[0],
             "test": test_mask.nonzero(as_tuple=True)[0], }
    return split

def FullTrainSplitter(dataset):
    text_g = dataset.data
    split = {"train": torch.arange(len(text_g.y))}
    return split


def CiteFSSplitter(dataset):
    labels = torch.tensor(dataset.data.y) if not isinstance(dataset.data.y, torch.Tensor) else dataset.data.y
    labels = labels.view(-1)
    cls_idx = []
    data_idx = []
    for i in range(labels.max() + 1):
        cls_idx.append(int(i))
        cls_data_idx = (labels == i).nonzero(as_tuple=True)[0]
        data_idx.append(cls_data_idx.numpy())
    cls_idx = np.array(cls_idx)
    return {k: [cls_idx, data_idx] for k in ["train", "valid", "test"]}


def CiteLinkSplitter(dataset):
    text_g = dataset.data
    edges = text_g.edge_index
    edge_perm = torch.randperm(len(edges[0]))
    train_offset = int(len(edge_perm) * 0.7)
    val_offset = int(len(edge_perm) * 0.8)
    edge_indices = {"train": edge_perm[:train_offset], "valid": edge_perm[train_offset:val_offset],
                    "test": edge_perm[val_offset:], }
    return edge_indices

def OfficialLinkSplitter(dataset):
    text_g = dataset.data
    train_idx = text_g.train_idx
    val_idx = text_g.val_idx
    test_idx = text_g.test_idx
    return {"train": train_idx, "valid": val_idx, "test": test_idx}






 

def KGSplitter(dataset):
    converted_triplet = dataset.get_idx_split()
    split = {}
    count = 0
    for name in converted_triplet:
        split[name] = torch.arange(count, count + len(converted_triplet[name][0]))
        count += len(converted_triplet[name][0])
    return split


def KGFSTrainSplitter(dataset):
    converted_triplet = dataset.get_idx_split()
    all_types = torch.cat([torch.tensor(converted_triplet[k][1]) for k in converted_triplet])
    with open("data/low_resource_split.json", "r") as f:
        lr_class_split = json.load(f)
    fs_split = []
    for split in lr_class_split[dataset.name]:
        cls_idx = []
        data_idx = []
        for cls in split:
            cls_idx.append(cls)
            cls_data_idx = (all_types == cls).nonzero(as_tuple=True)[0]
            data_idx.append(cls_data_idx.numpy())
        fs_split.append([np.array(cls_idx), data_idx])
    return {"train": fs_split[0], "valid": fs_split[1], "test": fs_split[2]}


def KGFSSplitter(dataset):
    converted_triplet = dataset.get_idx_split()
    all_types = {k: torch.tensor(converted_triplet[k][1]) for k in converted_triplet}
    offset = ([0] + [len(all_types[k]) for k in all_types])[:-1]
    for i in range(1, len(offset)):
        offset[i] += offset[i - 1]
    all_types_torch = torch.cat([all_types[k] for k in all_types])
    n_types = all_types_torch.max() + 1
    fs_split = {}
    for idx, name in enumerate(converted_triplet):
        cls_idx = []
        data_idx = []
        for i in range(n_types):
            cls_idx.append(i)
            cls_data_idx = (all_types[name] == i).nonzero(as_tuple=True)[0] + offset[idx]
            data_idx.append(cls_data_idx.numpy())
        fs_split[name] = [np.array(cls_idx), data_idx]
    return fs_split


def WikiSplitter(dataset):
    text_g = dataset.data
    wiki_split_idx = 0
    split = {"train": torch.where(text_g.train_mask[:, wiki_split_idx])[0].numpy(),
             "valid": torch.where(text_g.val_mask[:, wiki_split_idx])[0].numpy(),
             "test": torch.where(text_g.test_mask)[0].numpy(), }
    return split


def MolSplitter(dataset):
    return dataset.get_idx_split()

def TwentySplitter(dataset):
    text_g = dataset.data 
    labels = text_g.y.view(-1)
    splitter = LabelPerClassSplit(num_labels_per_class=20, num_valid=500, num_test=1000)
    train_mask, val_mask, test_mask = splitter(text_g, len(labels))
    split = {"train": train_mask.nonzero(as_tuple=True)[0],
                "valid": val_mask.nonzero(as_tuple=True)[0],
                "test": test_mask.nonzero(as_tuple=True)[0], }
    return split

def LowRateSplitter(dataset):
    text_g = dataset.data 
    labels = text_g.y.view(-1)
    if dataset.name == 'chemhiv':
        splits = dataset.get_idx_split()
        old_train = splits['train']
        val_mask = torch.tensor(splits['valid'])
        test_mask = torch.tensor(splits['test'])
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[old_train] = True
        train_mask_1 = train_mask & (labels == 1)
        train_mask_0 = train_mask & (labels == 0)
        train_mask_1 = train_mask_1.nonzero(as_tuple=True)[0][:3]
        train_mask_0 = train_mask_0.nonzero(as_tuple=True)[0][:3]
        train_mask = torch.cat([train_mask_0, train_mask_1])
        split = {"train": train_mask, "valid": val_mask, "test": test_mask}
    else:
        splitter = LabelPerClassSplit(num_labels_per_class=3, num_valid=500, num_test=5000)
        train_mask, val_mask, test_mask = splitter(text_g, len(labels))
        split = {"train": train_mask.nonzero(as_tuple=True)[0],
                "valid": val_mask.nonzero(as_tuple=True)[0],
                "test": test_mask.nonzero(as_tuple=True)[0], }
    return split

def MolFSTrainSplitter(dataset):
    fs_split = {}
    # use all chemblepre classes as training classes for few-shot/zero-shot tasks
    all_classes = dataset.y.view(len(dataset), -1)
    positive_samples = [(cls==1).nonzero(as_tuple=True)[0] for cls in all_classes.T]
    negative_samples = [(cls==0).nonzero(as_tuple=True)[0] for cls in all_classes.T]
    all_idx2cls = [np.array([i for i in range(2 * len(positive_samples))]), negative_samples + positive_samples]
    fs_split['train'] = all_idx2cls
    fs_split['valid'] = all_idx2cls
    fs_split['test'] = all_idx2cls

    return fs_split


#############################################
#   Preprocessing functions                 #
#############################################

def LinkConstructGraph(dataset, split):
    text_g = dataset.data
    edges = text_g.edge_index
    graph_dict = text_g.to_dict()
    graph_dict["edge_index"] = edges[:, split["train"]]
    train_graph = pyg.data.Data(**graph_dict)
    return train_graph


def KGConstructEdgeList(dataset, split):
    converted_triplet = dataset.get_idx_split()
    all_edges = torch.cat([torch.tensor(converted_triplet[k][0]) for k in converted_triplet], dim=0)
    all_types = torch.cat([torch.tensor(converted_triplet[k][1]) for k in converted_triplet])
    if len(split["train"]) == 2:
        idx = np.concatenate(split["train"][1])
    else:
        idx = split["train"]
    graph_dict = dataset.data.to_dict()
    graph_dict["edge_index"] = all_edges[idx].T
    graph_dict["edge_types"] = all_types[idx]
    graph = pyg.data.Data(**graph_dict)
    return all_edges, all_types, graph


def make_data(name, data, split_name, metric, eval_func, num_classes, **kwargs):
    # Wrap GraphTextDataset with DataWithMeta for easy evaluator construction
    return DataWithMeta(data, kwargs["batch_size"], sample_size=kwargs["sample_size"], metric=metric,
                        state_name=split_name + "_" + name, classes=num_classes,
                        meta_data={"eval_func": eval_func, "eval_mode": kwargs["eval_mode"]}, )


######################################################
#   Construct GraphTextDataset                       #
######################################################

def ConstructNodeCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, task_level, **kwargs):
    text_g = dataset.data
    global saved_edge_index
    if not kwargs.get("node_centered", True):
        text_g.edge_index = saved_edge_index[dataset.name].edge_index

    return SubgraphHierDataset(text_g, prompt_feats["class_node_text_feat"], prompt_feats["prompt_edge_text_feat"],
                               prompt_feats["noi_node_text_feat"], split[split_name], to_undirected=True,
                               process_label_func=to_bin_cls_func, prompt_edge_list=dataset.get_edge_list(task_level),  
                               **kwargs, )


def ConstructNodeNopromptCls(dataset, split, split_name, to_bin_cls_func, global_data, **kwargs):
    text_g = dataset.data

    return SubgraphNopromptDataset(text_g, text_g.label_text_feat, split[split_name], to_undirected=True,
                                   process_label_func=to_bin_cls_func, )


def ConstructLinkCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, task_level, **kwargs):
    text_g = dataset.data
    edges = text_g.edge_index
    train_graph = global_data
    global saved_edge_index
    saved_edge_index[dataset.name] = train_graph

    return SubgraphLinkHierDataset(train_graph, prompt_feats["class_node_text_feat"],
                                   prompt_feats["prompt_edge_text_feat"], prompt_feats["noi_node_text_feat"],
                                   edges.T[split[split_name]].numpy(), to_undirected=True, hop=3,
                                   process_label_func=to_bin_cls_func,
                                   prompt_edge_list=dataset.get_edge_list(task_level), **kwargs, )


def ConstructLinkNopromptCls(dataset, split, split_name, to_bin_cls_func, **kwargs):
    text_g = dataset.data
    edges = text_g.edge_index
    train_graph = kwargs["global_data"]

    return SubgraphNopromptLinkDataset(train_graph, train_graph.edge_label_feat, edges.T[split[split_name]].numpy(),
                                       prompt_feat=train_graph.prompt_text_edge_feat, to_undirected=True, hop=3,
                                       remove_edge=kwargs["remove_edge"], process_label_func=to_bin_cls_func,
                                       walk_length=kwargs["walk_length"], )


def ConstructKG(dataset, split, split_name, prompt_feats, to_bin_cls_func, task_level, global_data, **kwargs):
    edge_data = [global_data[0][split[split_name]].tolist(), global_data[1][split[split_name]].tolist()]

    return SubgraphKGHierDataset(global_data[-1], prompt_feats["class_node_text_feat"],
                                 prompt_feats["prompt_edge_text_feat"], prompt_feats["noi_node_text_feat"], edge_data,
                                 to_undirected=True, hop=2, process_label_func=to_bin_cls_func,
                                 prompt_edge_list=dataset.get_edge_list(task_level), **kwargs, )


def ConstructMolCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, task_level, global_data, **kwargs):
    return GraphListHierDataset(dataset, prompt_feats["class_node_text_feat"], prompt_feats["prompt_edge_text_feat"],
                                prompt_feats["noi_node_text_feat"], split[split_name],
                                process_label_func=to_bin_cls_func, prompt_edge_list=dataset.get_edge_list(task_level),
                                **kwargs, )


def ConstructMolNopromptCls(dataset, split, split_name, to_bin_cls_func, **kwargs):
    return GraphListNopromptDataset(dataset, dataset.label_text_feat, dataset.prompt_edge_feat, split[split_name],
                                    process_label_func=to_bin_cls_func, single_prompt_edge=True,
                                    walk_length=kwargs["walk_length"], )

def ConstructSegmentCls(dataset, split, split_name, prompt_feats, to_bin_cls_func, task_level, global_data, **kwargs):
    return MassiveDataset(dataset, prompt_feats["class_node_text_feat"], prompt_feats["prompt_edge_text_feat"],
                                prompt_feats["noi_node_text_feat"], split[split_name],
                                process_label_func=to_bin_cls_func, prompt_edge_list=dataset.get_edge_list(task_level),
                                **kwargs, )

def ConstructLowRateTask(dataset, split, split_name, prompt_feats, to_bin_cls_func, task_level, global_data, **kwargs):
    labels = dataset.y
    test_idx = split['test']
    original_idx = torch.arange(len(labels))
    pseudo_split = {"pseudo": original_idx}
    query_idx = []
    count = 0
    for d in split[split_name][1]:
        query_idx.append(torch.arange(count, count + len(d), dtype=torch.long))
        count += len(d)

    query_graph_dataset = globals()[kwargs["base_construct"]](dataset=dataset, split=pseudo_split, split_name="pseudo",prompt_feats=prompt_feats, to_bin_cls_func=None, global_data=global_data, task_level=task_level, **kwargs) 

    support_graph_dataset = globals()[kwargs["base_construct"]](dataset=dataset, split=pseudo_split, split_name="pseudo",prompt_feats=prompt_feats, to_bin_cls_func=None, global_data=global_data, task_level=task_level, **kwargs)

    fs_loader = LowRateLabelManager(split[split_name][0], query_idx, kwargs["k_shot"], 1, kwargs["n_way"], kwargs.get("min_k_shot"), kwargs.get("min_n_way"), test_idx, labels)

    return FewShotDataset(fs_loader, query_graph_dataset, support_graph_dataset, prompt_feats["prompt_edge_text_feat"][1:])

 



def ConstructFSTask(dataset, split, split_name, prompt_feats, to_bin_cls_func, global_data, task_level, **kwargs):
    ## here the split is n way k shot
    original_idx = np.concatenate(split[split_name][1])
    pseudo_split = {"pseudo": original_idx}
    query_idx = []
    count = 0
    for d in split[split_name][1]:
        query_idx.append(torch.arange(count, count + len(d), dtype=torch.long))
        count += len(d)

    query_graph_dataset = globals()[kwargs["base_construct"]](dataset=dataset, split=pseudo_split, split_name="pseudo",
                                                              prompt_feats=prompt_feats, to_bin_cls_func=None,
                                                              global_data=global_data, task_level=task_level, **kwargs)

    support_graph_dataset = globals()[kwargs["base_construct"]](dataset=dataset, split=pseudo_split,
                                                                split_name="pseudo", prompt_feats=prompt_feats,
                                                                to_bin_cls_func=None, global_data=global_data,
                                                                task_level=task_level, **kwargs)


    fs_loader = SimpleFSManager(split[split_name][0], query_idx, kwargs["k_shot"], 1, kwargs["n_way"],
                                kwargs.get("min_k_shot"), kwargs.get("min_n_way"))    
    return FewShotDataset(fs_loader, query_graph_dataset, support_graph_dataset,
                          prompt_feats["prompt_edge_text_feat"][1:])


####################################
#   process_label_function         #
####################################

def keep_label(embs, label):
    return label.long(), embs, None

def process_pth_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label.squeeze().to(torch.long)] = 1
    return label.view(1, -1).to(torch.long), embs, binary_rep


def process_reverse_binary_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label.squeeze().to(torch.long)] = 1
    embs = embs[[1, 0]]
    return label.view(1, -1).to(torch.long), embs, binary_rep


def process_multi_label(embs, label):
    valid_idx = label == label
    # valid_idx = torch.zeros_like(classes, dtype=torch.bool)
    return (
        torch.tensor([[0]]), embs[valid_idx.view(-1)].detach().clone(), label[:, valid_idx.view(-1)].detach().clone(),)


def process_positive_negative_multi_label(embs, label):
    valid_idx = label == label
    label = label[:, valid_idx.view(-1)].detach().clone()
    valid_idx = valid_idx.repeat(1, 2)
    label = torch.cat([label, 1 - label], dim=-1)

    return (torch.tensor([[0]]), embs[valid_idx.view(-1)].detach().clone(), label,)


def eval_process_label(embs, classes):
    return (torch.tensor([[0]]), embs, classes,)


def process_label_positive_only(embs, label):
    return torch.tensor([[0]]), embs[:len(label.view(-1))], label


def process_int_label(embs, label):
    binary_rep = torch.zeros((1, len(embs)))
    binary_rep[0, label] = 1
    return torch.tensor([label]).view(1, -1), embs, binary_rep


def hiv_trim_class(embs, label):
    one_hot_label = torch.nn.functional.one_hot(label.to(torch.long), num_classes=2)
    return label, embs, one_hot_label


def hiv_zs_class(embs, label):
    # one_hot_label = torch.nn.functional.one_hot(
    #     label.to(torch.long), num_classes=2
    # )
    return label, embs[0:1], label


def gen_can(n_class, label, size):
    can = torch.randint(n_class, size)
    mask = torch.rand(size) > 0.75
    can[mask] = label.view(-1)
    return can


def process_logic_label(embs, label):
    num_class = int(np.sqrt(len(embs) / 2))
    can = gen_can(num_class, label, (4, 2))
    or_label = ((can == label.view(-1)).sum(-1) > 0).to(torch.int)
    or_feat = embs[can[:, 0] * num_class + can[:, 1]]

    can = gen_can(num_class, label, (4, 2))
    and_label = ((can == label.view(-1)).sum(-1) == 0).to(torch.int)
    and_feat = embs[can[:, 0] * num_class + can[:, 1] + num_class ** 2]
    new_class_emb = torch.cat([or_feat, and_feat], dim=0)
    new_binary_rep = torch.cat([or_label, and_label]).view(1, -1)
    if isinstance(label, int):
        label = torch.tensor(label)
    return label.view(1, -1).to(torch.long), new_class_emb, new_binary_rep


none_process_label = None


class UnifiedTaskConstructor:
    def __init__(self, tasks: list[str], encoder: utils.SentenceEncoder, task_config_lookup: dict,
                 data_config_lookup: dict, root="cache_data", batch_size=256, sample_size=-1, node_centered = True, data_cache_path='cache_data_minilm'):
        """
        Construct tasks from a dictionary of dataset configurations. A task must contain a train dataset, but can
        have arbitrary number of valid/test dataset. A valid/test dataset is wrapped by a
        gp.lightning.data_template.DataWithMeta that contains information for evaluation metrics

        self.construct_exp construct all datasets.
        Args:
            tasks: a list of task names, they should be keys in the task_config_lookup
            encoder: utils.SentenceEncoder
            task_config_lookup: a dictionary for tasks, more details in Readme
            data_config_lookup: a dictionary for datasets construction in Readme
            root: dataset loading directory
            batch_size: int
            sample_size: int, -1 means full dataste
        """
        self.root = root
        self.tasks = tasks
        self.encoder = encoder
        self.task_config_lookup = task_config_lookup
        self.data_config_lookup = data_config_lookup
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.node_centered = node_centered
        self.data_cache_path = data_cache_path
        with open("data/low_resource_split.json", "r") as f:
            self.lr_class_split = json.load(f)

        self.dataset = {}  # keyed by base dataset names e.g. cora, pubmed and not cora-link
        self.dataset_split = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.preprocess_storage = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.datamanager = {}
        self.edges = {}
        self.datasets = {"train": [], "valid": [],
                         "test": []}  # train a list of Dataset, valid/test a list of DataWithMeta
        self.stage_names = {"train": [], "valid": [], "test": []}

    def construct_exp(self):
        val_task_index_lst = []
        val_pool_mode = []
        for task in self.tasks:
            config = self.task_config_lookup[task]
            config = copy.deepcopy(config)
            val_task_index_lst.append(self.construct_task(config))
            val_pool_mode.append(config["eval_pool_mode"])
        return val_task_index_lst, val_pool_mode

    def construct_task(self, config):
        """
        Datasets in a task are described in config["eval_set_constructs"] that describe the stage (train/valid/test)
        of the dataset.
        """
        val_task_index = []
        for stage_config in config["eval_set_constructs"]:
            if "dataset" not in stage_config:
                stage_config["dataset"] = config["dataset"]
            dataset_name = stage_config["dataset"]

            assert dataset_name in self.data_config_lookup, print(dataset_name)

            dataset_config = self.data_config_lookup[dataset_name]

            stage_ind = self.add_dataset(stage_config, dataset_config, full_name=config['dataset'])

            if stage_config["stage"] == "valid":
                val_task_index.append(stage_ind)
        return val_task_index

    def get_split_key(self, dataset_config):
        return dataset_config["dataset_name"] + "_" + dataset_config["task_level"]

    def get_stage_name(self, stage_config, dataset_config):
        return "_".join([stage_config["dataset"], self.get_split_key(dataset_config), stage_config["stage"],
                         stage_config["split_name"]])

    def get_ofa_data(self, dataset_config):
        dataset_name = dataset_config["dataset_name"]
        if dataset_name not in self.dataset:
            self.dataset[dataset_name] = name2dataset[dataset_name](dataset_name, root=self.root, encoder=self.encoder)
            ## change to undirected graph
        return self.dataset[dataset_name]

    def get_data_split(self, dataset_config):
        """
        Split data based on task_level
        """
        split_key = self.get_split_key(dataset_config)
        ## cache me if u can
        # if os.path.exists(os.path.join(self.data_cache_path, dataset_config["dataset_name"], f"{dataset_config['task_level']}_splits.pt")):
        #     split = torch.load(os.path.join(self.data_cache_path, dataset_config["dataset_name"], f"{dataset_config['task_level']}_splits.pt"))
        #     return split[split_key]
        if split_key not in self.dataset_split:
            dataset_splitter = dataset_config.get("dataset_splitter")
            split = globals()[dataset_splitter](
                self.dataset[dataset_config["dataset_name"]]) if dataset_splitter else None
            self.dataset_split[split_key] = split
        torch.save(self.dataset_split, os.path.join(self.data_cache_path, dataset_config["dataset_name"], f"{dataset_config['task_level']}_splits.pt"))
        return self.dataset_split[split_key]

    def get_global_data(self, dataset_config):
        """
        If global_data for a dataset is required, such as constructed train graph for link tasks, a preprocessing
        function is called and the returned values are stored.
        """
        split_key = self.get_split_key(dataset_config)
        # import ipdb; ipdb.set_trace()
        if split_key not in self.preprocess_storage:
            preprocessor = dataset_config.get("preprocess")
            global_data = globals()[preprocessor](self.dataset[dataset_config["dataset_name"]],
                                                  self.dataset_split[split_key]) if preprocessor else None
            self.preprocess_storage[split_key] = global_data
        return self.preprocess_storage[split_key]

    def add_dataset(self, stage_config, dataset_config, full_name):
        data = self.get_ofa_data(dataset_config)
        split = self.get_data_split(dataset_config)
        stage_name = self.get_stage_name(stage_config, dataset_config)
        # Evaluation datasets are constructed only once.
        if stage_config["stage"] != "train" and stage_name in self.stage_names[stage_config["stage"]]:
            return self.stage_names[stage_config["stage"]].index(stage_name)
        global_data = self.get_global_data(dataset_config)
        prompt_feats = data.get_prompt_text_feat(dataset_config["task_level"])
        data_name = stage_config["dataset"]
        data = globals()[dataset_config["construct"]](dataset=data, split=split, split_name=stage_config["split_name"],
                                                      prompt_feats=prompt_feats, to_bin_cls_func=globals()[
                dataset_config["process_label_func"]] if dataset_config.get("process_label_func") else None,
                                                      task_level=dataset_config["task_level"], global_data=global_data,
                                                      data_name = data_name, full_name = full_name, node_centered = self.node_centered,
                                                      **dataset_config["args"])
        if stage_config["stage"] == "train":
            self.datasets[stage_config["stage"]].append(data)
        else:
            eval_data = make_data(stage_config["dataset"], data, stage_config["split_name"],
                                  dataset_config["eval_metric"], globals()[dataset_config["eval_func"]],
                                  dataset_config["num_classes"], batch_size=self.batch_size,
                                  sample_size=self.sample_size, eval_mode=dataset_config["eval_mode"])
            self.datasets[stage_config["stage"]].append(eval_data)
        self.stage_names[stage_config["stage"]].append(stage_name)
        return self.stage_names[stage_config["stage"]].index(stage_name)

    def make_train_data(self, multiple, min_ratio, data_val_index=None):
        train_data = MultiDataset(self.datasets["train"], data_val_index=data_val_index, dataset_multiple=multiple,
                                  patience=3, window_size=5, min_ratio=min_ratio, )
        return train_data

    def make_full_dm_list(self, multiple, min_ratio, train_data=None):
        text_dataset = {
            "train": DataWithMeta(self.make_train_data(multiple, min_ratio) if not train_data else train_data,
                                  self.batch_size, sample_size=self.sample_size, ), "val": self.datasets["valid"],
            "test": self.datasets["test"], }
        return text_dataset
