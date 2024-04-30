from transformers import AutoConfig
import torch
from graphllm.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from torch_sparse import spspmm, spmm
from scipy.spatial.distance import cdist
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import scatter, mask_to_index, to_undirected, remove_self_loops, degree
from dgl.sampling import sample_neighbors
from collections import deque
import dgl
import torch.nn.functional as F
import os.path as osp
import os
from scipy.sparse import csr_array
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing
from tqdm import trange
import copy
import random
import json
import numpy as np


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def tokenizer_graph_token(prompt, tokenizer, graph_token_index=GRAPH_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_GRAPH_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [graph_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llaga' in config and 'llaga' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaGA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llaga")
            cfg.architectures[0] = 'LlagaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)




def normalize_adj(edge_index, num_nodes, edge_weight = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def _feature_similarity(features):
    return torch.from_numpy(1. - cdist(features, features, 'cosine'))


@torch.no_grad()
def get_feature_similarity(edge_index, features, k = 0):
    num_nodes = edge_index.shape[1]
    value = torch.ones(num_nodes)
    edge_index, value = normalize_adj(edge_index, num_nodes)
    if k == 0:
        ## directly use the feature matrix to compute the similarity
        return _feature_similarity(features.cpu())
    else:
        for _ in range(k - 1):
            edge_index, value = spspmm(edge_index, value, edge_index, value, num_nodes, num_nodes, num_nodes)
        aggr_features = spmm(edge_index, value, num_nodes, num_nodes, features)
        return _feature_similarity(aggr_features.cpu())

def dump_jsonl(data, json_filepath):
    path_name = osp.dirname(json_filepath)
    os.makedirs(path_name, exist_ok=True)
    with open(json_filepath, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')


def classification_prompt(category = 'paper', label_names = ['A', 'B'], gt = 0):
    num_of_classes = len(label_names)
    label_str = ", ".join(label_names)
    prompt = f"""
        Given a node-centered graph: <graph>, each node represents a {{DEFAULT_GRAPH_TOKEN}}, 
        we need to classify the center node into {num_of_classes}: {label_str}, please tell me which class the center node belongs to? 
    """ 
    human_conv = {
        "from": "human", 
        "value": prompt
    }
    gpt_conv = {
        "from": "gpt", 
        "value": f"{label_names[gt]}"
    }
    return human_conv, gpt_conv

def link_prediction_prompt(gt = 0):
    prompt = "Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."

    human_conv = {
        "from": "human", 
        "value": "cora_pos_edge" if gt == 1 else "cora_neg_edge"
    }

    gpt_conv = {
        "from": "gpt", 
        "value": "yes" if gt == 1 else "no"
    }

    return prompt, human_conv, gpt_conv


def get_mask(data, i = 0):
    """
        Given different types of mask format, return the first seed
    """
    if hasattr(data, 'train_mask'):
        if isinstance(data.train_mask, torch.Tensor):
            if data.train_mask.dim() == 1:
                return data.train_mask, data.val_mask, data.test_mask
            else:
                ## only for wikics
                return data.train_mask[:, i], data.val_mask[:, i], data.test_mask
        else:
            if i < len(data.train_mask):
                return data.train_mask[i], data.val_mask[i], data.test_mask[i]
            else:
                return data.train_mask[0], data.val_mask[0], data.test_mask[0]
    elif hasattr(data, 'train_masks'):
        if i < len(data.train_masks):
            return data.train_masks[i], data.val_masks[i], data.test_masks[i]
        else:
            return data.train_masks[0], data.val_masks[0], data.test_masks[0]


def generate_node_level_prompt_files(dataset_name, data):
    masks = get_mask(data)
    edge_list = generate_edge_list(data)
    for name, mask in zip(["train", "val", "test"], masks): 
        filename = osp.join(dataset_name, f"sampled_2_10_{name}.jsonl")
        jsonl_list = []
        ### 4 properties: 
        ### 1. id (of the center node)
        all_idxs = mask_to_index(mask)
        ### 2. graph (sequence of the generated tree)
        for idx in all_idxs:
            tree = {}
            tree['id'] = idx.item()
            tree['graph'] = get_fix_shape_subgraph_sequence_fast(edge_list, idx.item(), 2, 10)
            jsonl_list.append(tree)
        ### 3 should be generated dynamically to save storage
        ### 3. conversation (first element human-info, second element gpt-info)
        dump_jsonl(jsonl_list, filename)

def generate_edge_level_prompt_files(dataset_name, data):
    """
        Here, the data should be the test data object (assuming using randomlinksplit)
    """
    train_idx = data.train_idx
    val_idx = data.val_idx
    test_idx = data.test_idx
    edge_index = data.edge_index
    orig_edge_index = data.edge_index
    ## remove all test_idx to prevent leakage
    edge_index = edge_index[:, train_idx]
    data.edge_index = edge_index
    edge_list = generate_edge_list(data)
    edge_index = orig_edge_index

    train_len, val_len, test_len = len(train_idx), len(val_idx), len(test_idx)
    pos_index = train_len + val_len + test_len
    num_nodes = data.node_text_feat.shape[0]
    neg_edge_idx = generate_negative_samples(edge_index, num_nodes, pos_index)

    for split_name, idx in zip(['train', 'test'], [train_idx, test_idx]):
        name = osp.join(dataset_name, f"edge_sampled_2_10_only_{split_name}.jsonl")
        ## positive edges
        pos_edges = edge_index[:, idx]
        ## negative edges
        neg_edges = neg_edge_idx[:, idx]

        jsonl_list = []
        for i in range(idx.shape[0]):
            tree = {}
            left, right = pos_edges[:, i]
            tree['id'] = [left.item(), right.item()]
            left_g = get_fix_shape_subgraph_sequence_fast(edge_list, left.item(), 2, 10, avoid_idx=right.item())
            right_g = get_fix_shape_subgraph_sequence_fast(edge_list, right.item(), 2, 10, avoid_idx=left.item())
            tree['graph'] = [left_g, right_g]
            tree['gt'] = 1
            jsonl_list.append(tree)

            tree = {}
            left, right = neg_edges[:, i]
            tree['id'] = [left.item(), right.item()]
            left_g = get_fix_shape_subgraph_sequence_fast(edge_list, left.item(), 2, 10, avoid_idx=right.item())
            right_g = get_fix_shape_subgraph_sequence_fast(edge_list, right.item(), 2, 10, avoid_idx=left.item())
            tree['graph'] = [left_g, right_g]
            tree['gt'] = 0
            jsonl_list.append(tree)
        dump_jsonl(jsonl_list, name)

def generate_negative_samples(edge_index, num_nodes, pos_index):
    adj = csr_array((torch.ones(len(edge_index[0])), (edge_index[0], edge_index[1]),), shape=(num_nodes, num_nodes), )
    dense_adj = adj.todense() == 0
    neg_row, neg_col = np.nonzero(dense_adj)
    neg_edge_idx = np.random.permutation(len(neg_row))[:pos_index]
    neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
    neg_edges = np.stack([neg_row, neg_col], axis=1)
    return torch.tensor(neg_edges).t()


def generate_edge_list(data):
    """
        Turn edge index into adjacency list
    """
    ## must first turn into undirected graph
    ## and no self loop
    edge_index = to_undirected(data.edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()
    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    return edge_list


def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def partition_propagate(self, data_edge_index, x, norm, select_idx=None, chunk_size=800, cuda=False):
        if select_idx is None:
            n = x.shape[0]
            select_idx = torch.arange(n)
        else:
            n = select_idx.shape[0]

        os=[]
        for i in trange(0, n, chunk_size):
            key=select_idx[i:i+chunk_size]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(key, 1, data_edge_index, relabel_nodes=True)
            if cuda:
                o =  self.propagate(edge_index.cuda(), x=x[subset].cuda(), norm=norm[edge_mask].cuda())
            else:
                o = self.propagate(edge_index, x=x[subset], norm=norm[edge_mask])
            os.append(o[mapping])

        return torch.cat(os, dim=0)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def generate_multi_hop_x(data, x, full_path, emb="sbert"):
    ## this is only used for node, for link we generate on the fly 
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mp = MP()
    torch.save(x, osp.join(full_path, f"{emb}_0hop_x.pt"))
    for i in range(4):
        x = mp.partition_propagate(edge_index, x=x, norm=norm, chunk_size=200, cuda=True)
        torch.save(x.cpu(), osp.join(full_path, f"{emb}_{i + 1}hop_x.pt"))
        




if __name__ == '__main__':
    ### Unit test
    pass 
