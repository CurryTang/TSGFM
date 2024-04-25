"""
utility functions and global variables
"""

import os
from distutils.util import strtobool
from math import inf

import torch
import numpy as np
"""
hitrate@k, mean reciprocal rank (MRR) and Area under the receiver operator characteristic curve (AUC) evaluation metrics
"""
from sklearn.metrics import roc_auc_score

"""
utils for getting the largest connected component of a graph
"""
from torch_geometric.data import Data, InMemoryDataset

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_DIC = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
               'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
               'dynamic_val': True, 'model': 'hashing', 'sign_k': 2,
               'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1, 'use_feature': True, 'dropout': 0,
               'label_dropout': 0, 'feature_dropout': 0,
               'add_normed_features': False, 'use_RA': False, 'hidden_channels': 32, 'load_features': True,
               'load_hashes': True, 'use_zero_one': True, 'wandb': False, 'batch_size': 32, 'num_workers': 1,
               'cache_subgraph_features': False, 'eval_batch_size': 1000, 'num_negs': 1}


def evaluate_hits(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,
                  Ks=[20, 50, 100], use_val_negs_for_train=True):
    """
    Evaluate the hit rate at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic[ks]
    """
    results = {}
    # As the training performance is used to assess overfitting it can help to use the same set of negs for
    # train and val comparisons.
    if use_val_negs_for_train:
        neg_train = neg_val_pred
    else:
        neg_train = neg_train_pred
    for K in Ks:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """
    Evaluate the mean reciprocal rank at K
    :param evaluator: an ogb Evaluator object
    :param pos_val_pred: Tensor[val edges]
    :param neg_val_pred: Tensor[neg val edges]
    :param pos_test_pred: Tensor[test edges]
    :param neg_test_pred: Tensor[neg test edges]
    :param Ks: top ks to evaluatate for
    :return: dic with single key 'MRR'
    """
    neg_train_pred = neg_train_pred.view(pos_train_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        # for mrr negs all have the same src, so can't use the val negs, but as long as the same  number of negs / pos are
        # used the results will be comparable.
        'y_pred_neg': neg_train_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)

    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    """
    the ROC AUC
    :param val_pred: Tensor[val edges] predictions
    :param val_true: Tensor[val edges] labels
    :param test_pred: Tensor[test edges] predictions
    :param test_true: Tensor[test edges] labels
    :return:
    """
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results





def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


def get_num_samples(sample_arg, dataset_len):
    """
    convert a sample arg that can be a number of % into a number of samples
    :param sample_arg: float interpreted as % if < 1 or count if >= 1
    :param dataset_len: the number of data points before sampling
    :return:
    """
    if sample_arg < 1:
        samples = int(sample_arg * dataset_len)
    else:
        samples = int(min(sample_arg, dataset_len))
    return samples


def select_embedding(args, num_nodes, device):
    """
    select a node embedding. Used by SEAL models (the E in SEAL is for Embedding)
    and needed for ogb-ddi where there are no node features
    :param args: Namespace of cmd args
    :param num_nodes: Int number of nodes to produce embeddings for
    :param device: cpu or cuda
    :return: Torch.nn.Embedding [n_nodes, args.hidden_channels]
    """
    if args.train_node_embedding:
        emb = torch.nn.Embedding(num_nodes, args.hidden_channels).to(device)
    elif args.pretrained_node_embedding:
        weight = torch.load(args.pretrained_node_embedding)
        emb = torch.nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = False
    else:
        emb = None
    return emb


def get_pos_neg_edges(data, sample_frac=1):
    """
    extract the positive and negative supervision edges (as opposed to message passing edges) from data that has been
     transformed by RandomLinkSplit
    :param data: A train, val or test split returned by RandomLinkSplit
    :return: positive edge_index, negative edge_index.
    """
    device = data.edge_index.device
    edge_index = data['edge_label_index'].to(device)
    labels = data['edge_label'].to(device)
    pos_edges = edge_index[:, labels == 1].t()
    neg_edges = edge_index[:, labels == 0].t()
    if sample_frac != 1:
        n_pos = pos_edges.shape[0]
        np.random.seed(123)
        perm = np.random.permutation(n_pos)
        perm = perm[:int(sample_frac * n_pos)]
        pos_edges = pos_edges[perm, :]
        neg_edges = neg_edges[perm, :]
    return pos_edges.to(device), neg_edges.to(device)


def get_same_source_negs(num_nodes, num_negs_per_pos, pos_edge):
    """
    The ogb-citation datasets uses negatives with the same src, but different dst to the positives
    :param num_nodes: Int node count
    :param num_negs_per_pos: Int
    :param pos_edge: Int Tensor[2, edges]
    :return: Int Tensor[2, edges]
    """
    print(f'generating {num_negs_per_pos} single source negatives for each positive source node')
    dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
    src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
    return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)


def neighbors(fringe, A, outgoing=True):
    """
    Retrieve neighbours of nodes within the fringe
    :param fringe: set of node IDs
    :param A: scipy CSR sparse adjacency matrix
    :param outgoing: bool
    :return:
    """
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def get_src_dst_degree(src, dst, A, max_nodes):
    """
    Assumes undirected, unweighted graph
    :param src: Int Tensor[edges]
    :param dst: Int Tensor[edges]
    :param A: scipy CSR adjacency matrix
    :param max_nodes: cap on max node degree
    :return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree


def str2bool(x):
    """
    hack to allow wandb to tune boolean cmd args
    :param x: str of bool
    :return: bool
    """
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')