from transformers import AutoConfig
import torch
from graphllm.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN
from torch_sparse import spspmm, spmm
from scipy.spatial.distance import cdist
from torch_geometric.utils import remove_self_loops
<<<<<<< HEAD
from torch_geometric.utils import scatter, mask_to_index
from dgl.sampling import sample_neighbors
import dgl
=======
from torch_geometric.utils import scatter
>>>>>>> 445f2f0e39317634a16f215cf6a7756fb6ab93cb

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

@torch.no_grad()
def get_propagated_features(edge_index, features, edge_attr = None, k = 0, normalize = True):
    """
        Input: edge_index, features
        Output: a list of propagated features
        k = 0 <-> center node
        k = 1 <-> 1-hop neighbors
    """
    # import ipdb; ipdb.set_trace()
    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]
    if edge_attr is None:
        value = torch.ones(num_edges)
    else:
        value = edge_attr
    if normalize:
        edge_index, value = normalize_adj(edge_index, num_nodes)
    results = []
    if k == 0:
        # features = features.reshape(num_nodes, 1, -1)
        results.append(features)
        return results
    else:
        results.append(features)
        aggr_features = features
        # import ipdb; ipdb.set_trace()
        for _ in range(k):
            # edge_index, value = spspmm(edge_index, value, edge_index, value, num_nodes, num_nodes, num_nodes)
            aggr_features = spmm(edge_index, value, num_nodes, num_nodes, aggr_features)
            results.append(aggr_features)
        return results
<<<<<<< HEAD

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


def generate_node_level_prompt_files(dataset_name, data):
    masks = [
        data.train_mask,
        data.val_mask,
        data.test_mask
    ]
    jsonl_list = []
    for name, mask in zip(["train", "val", "test"], masks): 
        filename = f"sampled_2_10_{name}.jsonl"
        ### 4 properties: 
        ### 1. id (of the center node)
        all_idxs = mask_to_index(mask)
        ### 2. graph (sequence of the generated tree)
        
        ### 3 should be generated dynamically to save storage
        ### 3. conversation (first element human-info, second element gpt-info)


def generate_edge_level_prompt_files(dataset_name, data):
    pass 


def generate_tree(data, hop = 2, neighbor = 10):
    edge_index = data.edge_index
    edges = (edge_index[0], edge_index[1])
    dgl_graph = dgl.graph(edges)
    


if __name__ == '__main__':
    ### Unit test
    pass 
=======
>>>>>>> 445f2f0e39317634a16f215cf6a7756fb6ab93cb
