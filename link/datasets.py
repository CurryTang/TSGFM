"""
Code based on
https://github.com/facebookresearch/SEAL_OGB
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SEAL reformulates link prediction as a subgraph classification problem. To do this subgraph datasets must first be constructed
"""

from math import inf
import random
import os
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops)
from torch_sparse import coalesce
from tqdm import tqdm
import scipy.sparse as ssp

from link.utils import get_src_dst_degree, neighbors, get_pos_neg_edges
from link.labelling_tricks import drnl_node_labeling, de_node_labeling, de_plus_node_labeling

from link.heuristics import RA 
from link.utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
from models.hashing import ElphHashes
from time import time
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
import torch_sparse


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None):
        self.data = data
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 1.:
            name = f'SEAL_{self.split}_data'
        else:
            name = f'SEAL_{self.split}_data_{self.percent}'
        name += '.pt'
        return [name]

    def process(self):

        if self.use_coalesce:  # compress multi-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            self.pos_edges, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            self.neg_edges, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.max_dist, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, pos_edges, neg_edges, num_hops, percent=1., use_coalesce=False, node_label='drnl',
                 ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 **kwargs):
        self.data = data
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        super(SEALDynamicDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0).tolist()
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, self.max_nodes_per_hop)
        if self.sign:
            x = [self.data.x]
            x += [self.data[f'x{i}'] for i in range(1, self.k + 1)]
        else:
            x = self.data.x
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label, self.max_dist, src_degree, dst_degree)

        return data


def sample_data(data, sample_arg):
    if sample_arg <= 1:
        samples = int(sample_arg * len(data))
    elif sample_arg != inf:
        samples = int(sample_arg)
    else:
        samples = len(data)
    if samples != inf:
        sample_indices = torch.randperm(len(data))[:samples]
    return data[sample_indices]


def get_train_val_test_datasets(root, train_data, val_data, test_data, args, name):
    sample = 'all'
    if args.node_label == 'drnl':
        path = f'{root}_seal_{sample}_hops_{args.num_hops}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}_{name}'
    else:
        path = f'{root}_seal_{sample}_hops_{args.num_hops}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}_{args.node_label}_{name}'
    print(f'seal data path: {path}')
    use_coalesce = False
    # get percents used only for naming the SEAL dataset files and caching
    train_percent, val_percent, test_percent = 1 - (args.val_pct + args.test_pct), args.val_pct, args.test_pct
    # probably should be an attribute of the dataset and not hardcoded
    directed = False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    pos_train_edge = sample_data(pos_train_edge, args.train_samples)
    neg_train_edge = sample_data(neg_train_edge, args.train_samples)
    pos_val_edge = sample_data(pos_val_edge, args.val_samples)
    neg_val_edge = sample_data(neg_val_edge, args.val_samples)
    pos_test_edge = sample_data(pos_test_edge, args.test_samples)
    neg_test_edge = sample_data(neg_test_edge, args.test_samples)

    print(
        f'after sampling, using {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    dataset_class = 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path,
        train_data,
        pos_train_edge,
        neg_train_edge,
        num_hops=args.num_hops,
        percent=train_percent,
        split='train',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.model == 'sign',
        k=args.sign_k
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path,
        val_data,
        pos_val_edge,
        neg_val_edge,
        num_hops=args.num_hops,
        percent=val_percent,
        split='valid',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.model == 'sign',
        k=args.sign_k
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path,
        test_data,
        pos_test_edge,
        neg_test_edge,
        num_hops=args.num_hops,
        percent=test_percent,
        split='test',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.model == 'sign',
        k=args.sign_k
    )
    return train_dataset, val_dataset, test_dataset


def get_seal_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    """
    Extract the k-hop enclosing subgraph around link (src, dst) from A.
    it permutes the node indices so the returned subgraphs are not immediately recognisable as subgraphs and it is not
    parallelised.
    For directed graphs it adds both incoming and outgoing edges in the BFS equally and then for the target edge src->dst
    it will also delete any dst->src edge, it's unclear if this is a feature or a bug.
    :param src: source node for the edge
    :param dst: destination node for the edge
    :param num_hops:
    :param A:
    :param sample_ratio: This will sample down the total number of neighbours (from both src and dst) at each hop
    :param max_nodes_per_hop: This will sample down the total number of neighbours (from both src and dst) at each hop
                            can be used in conjunction with sample_ratio
    :param node_features:
    :param y:
    :param directed:
    :param A_csc:
    :return:
    """
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for hop in range(1, num_hops + 1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [hop] * len(fringe)
    # this will permute the rows and columns of the input graph and so the features must also be permuted
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph. Works as the first two elements of nodes are the src and dst node
    # this can throw warnings as csr sparse matrices aren't efficient for removing edges, but these graphs are quite sml
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if isinstance(node_features, list):
        node_features = [feat[nodes] for feat in node_features]
    elif node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', max_dist=1000, src_degree=None,
                        dst_degree=None):
    """
    Constructs a pyg graph for this subgraph and adds an attribute z containing the node_label
    @param node_ids: list of node IDs in the subgraph
    @param adj: scipy sparse CSR adjacency matrix
    @param dists: an n_nodes list containing shortest distance (in hops) to the src or dst node
    @param node_features: The input node features corresponding to nodes in node_ids
    @param y: scalar, 1 if positive edges, 0 if negative edges
    @param node_label: method to add the z attribute to nodes
    @return:
    """
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1, max_dist)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes, src_degree=src_degree, dst_degree=dst_degree)
    return data


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000,
                                directed=False, A_csc=None):
    """
    Extract a num_hops subgraph around every edge in the link index
    @param link_index: positive or negative supervision edges from train, val or test
    @param A: A scipy sparse CSR matrix containing the message passing edge
    @param x: features on the data
    @param y: 1 for positive edges, 0 for negative edges
    @param num_hops: the number of hops from the src or dst node to expand the subgraph to
    @param node_label:
    @param ratio_per_hop:
    @param max_nodes_per_hop:
    @param directed:
    @param A_csc: None if undirected, otherwise converts to a CSC matrix
    @return:
    """
    data_list = []
    for src, dst in tqdm(link_index.tolist()):
        src_degree, dst_degree = get_src_dst_degree(src, dst, A, max_nodes_per_hop)
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label, max_dist, src_degree, dst_degree)
        data_list.append(data)

    return data_list



class HashDataset(Dataset):
    """
    A class that combines propagated node features x, and subgraph features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, use_coalesce=False,
            directed=False, name="cora", **kwargs):
        if args.model != 'ELPH':  # elph stores the hashes directly in the model class for message passing
            self.elph_hashes = ElphHashes(args)  # object for hash and subgraph feature operations
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.load_features = args.load_features
        self.load_hashes = args.load_hashes
        self.use_zero_one = args.use_zero_one
        self.cache_subgraph_features = args.cache_subgraph_features
        self.max_hash_hops = args.max_hash_hops
        self.use_feature = args.use_feature
        self.use_RA = args.use_RA
        self.hll_p = args.hll_p
        self.subgraph_features = None
        self.hashes = None
        self.name = name
        super(HashDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        if self.directed:  # make undirected graphs like citation2 directed
            # print(
            #     f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
            self.edge_index, self.edge_weight = to_undirected(data.edge_index, self.edge_weight)
        else:
            self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

        if self.use_RA:
            self.RA = RA(self.A, self.links, batch_size=2000000)[0]

        if args.model == 'ELPH':  # features propagated in the model instead of preprocessed
            self.x = data.x
        else:
            self.x = self._preprocess_node_features(data, self.edge_index, self.edge_weight, args.sign_k)
            # ELPH does hashing and feature prop on the fly
            # either set self.hashes or self.subgraph_features depending on cmd args
            self._preprocess_subgraph_features(self.edge_index.device, data.num_nodes, args.num_negs)

    def _generate_sign_features(self, data, edge_index, edge_weight, sign_k):
        """
        Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
         https://arxiv.org/abs/2004.11198
        @param data: A pyg data object
        @param sign_k: the maximum number of times to apply the propagation operator
        @return:
        """
        try:
            num_nodes = data.x.size(0)
        except AttributeError:
            num_nodes = data.num_nodes
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight.float(), num_nodes)
        if sign_k == 0:
            # for most datasets it works best do one step of propagation
            xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        else:
            xs = [data.x]
            for _ in range(sign_k):
                x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
                xs.append(x)
            xs = torch.cat(xs, dim=-1)
        return xs

    def _preprocess_node_features(self, data, edge_index, edge_weight, sign_k=0):
        """
        preprocess the node features
        @param data: pyg Data object
        @param edge_weight: pyg edge index Int Tensor [edges, 2]
        @param sign_k: the number of propagation steps used by SIGN
        @return: Float Tensor [num_nodes, hidden_dim]
        """
        if sign_k == 0:
            feature_name = f'{self.root}_{self.split}_featurecache_{self.name}_.pt'
        else:
            feature_name = f'{self.root}_{self.split}_k{sign_k}_featurecache_{self.name}_.pt'
        if self.load_features and os.path.exists(feature_name):
            #print('loading node features from disk')
            x = torch.load(feature_name).to(edge_index.device)
        else:
            #print('constructing node features')
            start_time = time()
            x = self._generate_sign_features(data, edge_index, edge_weight, sign_k)
            #print("Preprocessed features in: {:.2f} seconds".format(time() - start_time))
            if self.load_features:
                torch.save(x.cpu(), feature_name)
        return x

    def _read_subgraph_features(self, name, device):
        """
        return True if the subgraph features can be read off disk, otherwise returns False
        @param name:
        @param device:
        @return:
        """
        retval = False
        # look on disk
        if self.cache_subgraph_features and os.path.exists(name):
            #print(f'looking for subgraph features in {name}')
            self.subgraph_features = torch.load(name).to(device)
            #print(f"cached subgraph features found at: {name}")
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are inconsistent with the link object. Delete subgraph features file and regenerate'
            retval = True
        return retval

    def _generate_file_names(self, num_negs):
        """
        get the subgraph feature file name and the stubs needed to make a new one if necessary
        :param num_negs: Int negative samples / positive sample
        :return:
        """
        if self.max_hash_hops != 2:
            hop_str = f'{self.max_hash_hops}hop_'
        else:
            hop_str = ''
        end_str = f'_{hop_str}subgraph_featurecache.pt'
        year_str = ''
        if num_negs == 1 or self.split != 'train':
            subgraph_cache_name = f'{self.root}{self.split}{year_str}{end_str}_{self.name}_'
        else:
            subgraph_cache_name = f'{self.root}{self.split}_negs{num_negs}{year_str}{end_str}_{self.name}_'
        return subgraph_cache_name, year_str, hop_str

    def _preprocess_subgraph_features(self, device, num_nodes, num_negs=1):
        """
        Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
        Sets self.subgraph_features
        @return:
        """
        subgraph_cache_name, year_str, hop_str = self._generate_file_names(num_negs)
        found_subgraph_features = self._read_subgraph_features(subgraph_cache_name, device)
        if not found_subgraph_features:
            # if self.cache_subgraph_features:
            #     print(f'no subgraph features found at {subgraph_cache_name}')
            # print('generating subgraph features')
            hash_name = f'{self.root}{self.split}{year_str}_{hop_str}hashcache{self.name}_.pt'
            cards_name = f'{self.root}{self.split}{year_str}_{hop_str}cardcache{self.name}_.pt'
            if self.load_hashes and os.path.exists(hash_name):
                #print('loading hashes from disk')
                hashes = torch.load(hash_name)
                if os.path.exists(cards_name):
                    #print('loading cards from disk')
                    cards = torch.load(cards_name)
                else:
                    print(f'hashes found at {hash_name}, but cards not found. Delete hashes and run again')
            else:
                #print('no hashes found on disk, constructing hashes...')
                start_time = time()
                hashes, cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
                #print("Preprocessed hashes in: {:.2f} seconds".format(time() - start_time))
                if self.load_hashes:
                    torch.save(cards, cards_name)
                    torch.save(hashes, hash_name)
            # print('constructing subgraph features')
            start_time = time()
            self.subgraph_features = self.elph_hashes.get_subgraph_features(self.links, hashes, cards,
                                                                            self.args.subgraph_feature_batch_size)
            # print("Preprocessed subgraph features in: {:.2f} seconds".format(time() - start_time))
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are a different shape link object. Delete subgraph features file and regenerate'
            if self.cache_subgraph_features:
                torch.save(self.subgraph_features, subgraph_cache_name)
        if self.args.floor_sf and self.subgraph_features is not None:
            self.subgraph_features[self.subgraph_features < 0] = 0
            # print(
            #     f'setting {torch.sum(self.subgraph_features[self.subgraph_features < 0]).item()} negative values to zero')
        if not self.use_zero_one and self.subgraph_features is not None:  # knock out the zero_one features (0,1) and (1,0)
            if self.max_hash_hops > 1:
                self.subgraph_features[:, [4, 5]] = 0
            if self.max_hash_hops == 3:
                self.subgraph_features[:, [11, 12]] = 0  # also need to get rid of (0, 2) and (2, 0)

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            subgraph_features = self.subgraph_features[idx]
        else:
            subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        if self.use_RA:
            RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
            RA = torch.tensor([RA], dtype=torch.float)
        else:
            RA = -1
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return subgraph_features, node_features, src_degree, dst_degree, RA, y


def get_hashed_train_val_test_datasets(root, train_data, val_data, test_data, args, name, directed=False):
    root = f'{root}/elph_'
    #print(f'data path: {root}')
    use_coalesce = False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    # print(
    #     f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
    #     f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
    #     f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    # print('constructing training dataset object')
    train_dataset = HashDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                use_coalesce=use_coalesce, directed=directed, name=name)
    #print('constructing validation dataset object')
    val_dataset = HashDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                              use_coalesce=use_coalesce, directed=directed, name=name)
    #print('constructing test dataset object')
    test_dataset = HashDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                               use_coalesce=use_coalesce, directed=directed, name=name)
    return train_dataset, val_dataset, test_dataset

# def get_normal_train_val_test_datasets(root, train_data, val_data, test_data, args, directed=False):
#     ## root is not used
#     pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
#     pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
#     pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)




class HashedTrainEvalDataset(Dataset):
    """
    Subset of the full training dataset used to get unbiased estimate of training performance for large datasets
    where otherwise training eval is a significant % of runtime
    """

    def __init__(
            self, links, labels, subgraph_features, RA, dataset):
        super(HashedTrainEvalDataset, self).__init__()
        self.links = links
        self.labels = labels
        self.edge_index = dataset.edge_index
        self.subgraph_features = subgraph_features
        self.x = dataset.x
        self.degrees = dataset.degrees
        self.RA = RA

    def len(self):
        return len(self.links)

    def get(self, idx):
        return self.links[idx]


def make_train_eval_data(args, train_dataset, num_nodes, n_pos_samples=5000, negs_per_pos=1000):
    """
    A much smaller subset of the training data to get a comparable (with test and val) measure of training performance
    to diagnose overfitting
    @param args: Namespace object of cmd args
    @param train_dataset: pyG Dataset object
    @param n_pos_samples: The number of positive samples to evaluate the training set on
    @return: HashedTrainEvalDataset
    """
    # ideally the negatives and the subgraph features are cached and just read from disk
    # need to save train_eval_negs_5000 and train_eval_subgraph_features_5000 files
    # and ensure that the order is always the same just as with the other datasets
    print('constructing dataset to evaluate training performance')
    dataset_name = args.dataset_name
    pos_sample = train_dataset.pos_edges[:n_pos_samples]  # [num_edges, 2]
    neg_sample = train_dataset.neg_edges[:n_pos_samples * negs_per_pos]  # [num_neg_edges, 2]
    assert torch.all(torch.eq(pos_sample[:, 0].repeat_interleave(negs_per_pos), neg_sample[:,
                                                                                0])), 'negatives have different source nodes to positives. Delete train_eval_negative_samples_* and subgraph features and regenerate'
    links = torch.cat([pos_sample, neg_sample], 0)  # [n_edges, 2]
    labels = [1] * pos_sample.size(0) + [0] * neg_sample.size(0)
    if train_dataset.use_RA:
        pos_RA = train_dataset.RA[:n_pos_samples]
        neg_RA = RA(train_dataset.A, neg_sample, batch_size=2000000)[0]
        RA_links = torch.cat([pos_RA, neg_RA], dim=0)
    else:
        RA_links = None
    pos_sf = train_dataset.subgraph_features[:n_pos_samples]
    n_pos_edges = len(train_dataset.pos_edges)
    neg_sf = train_dataset.subgraph_features[n_pos_edges: n_pos_edges + len(neg_sample)]
    # check these indices are all negative samples
    assert sum(train_dataset.labels[n_pos_edges: n_pos_edges + len(neg_sample)]) == 0
    subgraph_features = torch.cat([pos_sf, neg_sf], dim=0)
    train_eval_dataset = HashedTrainEvalDataset(links, labels, subgraph_features, RA_links, train_dataset)
    return train_eval_dataset