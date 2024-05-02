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
import random
from typing import Optional, Tuple, Union
import numpy as np
from torch import Tensor
from torch_geometric.utils import coalesce, cumsum, degree, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ['sparse', 'dense']

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(edge_index, size, bipartite,
                                           force_undirected)

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    if method == 'dense':
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx = idx.to('cpu')
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device='cpu')
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to('cpu'))
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = rnd[~mask].to(edge_index.device)
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        ptr = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_sampling(edge_index, num_nodes[i],
                                           num_neg_samples, method,
                                           force_undirected)
        neg_edge_index += ptr[i]
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


def structured_negative_sampling(edge_index, num_nodes: Optional[int] = None,
                                 contains_neg_self_loops: bool = True):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:

        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    r"""Returns :obj:`True` if
    :meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
    if atleast one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:

        >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # True if there exists no node that is connected to all other nodes.
    return bool(torch.all(deg < max_num_neighbors))


###############################################################################


def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


def edge_index_to_vector(
    edge_index: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[Tensor, int]:

    row, col = edge_index

    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = torch.arange(1, num_nodes, device=row.device).cumsum(0)[row]
        idx = row.mul_(num_nodes).add_(col).sub_(offset)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.mul_(num_nodes - 1).add_(col)
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(idx: Tensor, size: Tuple[int, int], bipartite: bool,
                         force_undirected: bool = False) -> Tensor:

    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode='floor')
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = torch.arange(1, num_nodes, device=idx.device).cumsum(0)
        end = torch.arange(num_nodes, num_nodes * num_nodes, num_nodes,
                           device=idx.device)
        row = torch.bucketize(idx, end.sub_(offset), right=True)
        col = offset[row].add_(idx) % num_nodes
        return torch.stack([torch.cat([row, col]), torch.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode='floor')
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return torch.stack([row, col], dim=0)




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
    
    for K in [20, 50, 100]:
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

        print(f'Hits@{K}: Train: {train_hits:.4f}, Val: {valid_hits:.4f}, Test: {test_hits:.4f}')

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