import os.path as osp
import torch
from graphllm.utils import generate_node_level_prompt_files, generate_edge_level_prompt_files, generate_multi_hop_x
## the following is the full set
# node_level_datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookchild', 'bookhis', 'elecomp', 'elephoto', 'sportsfit', 'products']

def load_local_data(dataset_name):
    path = "/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/cache_data_minilm"
    return torch.load(osp.join(path, dataset_name, 'processed', 'geometric_data_processed.pt'))[0]

def cite_link_splitter(data):
    edges = data.edge_index
    edge_perm = torch.randperm(len(edges[0]))
    train_offset = int(len(edge_perm) * 0.7)
    val_offset = int(len(edge_perm) * 0.8)
    edge_indices = {"train": edge_perm[:train_offset], "valid": edge_perm[train_offset:val_offset],
                    "test": edge_perm[val_offset:], }
    return edge_indices


if __name__ == '__main__':
    preprocessed_data_path = '/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/cache_data_minilm'
    ## we first sanity check the simplest one
    node_level_datasets = ['cora']
    link_level_datasets = ['cora']
    # link_level_datasets = ['citeseer']

    for dataset_name in node_level_datasets:
        node_data = load_local_data(dataset_name)
        full_name = '/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/outdata'
        full_name = osp.join(full_name, dataset_name)
        generate_node_level_prompt_files(full_name, node_data)
        print(f'Node level data for {dataset_name} has been generated')

        generate_multi_hop_x(node_data, node_data.node_text_feat, full_name)
        print(f'Multi-hop data for {dataset_name} has been generated')

    for dataset_name in link_level_datasets:
        link_data = load_local_data(dataset_name)
        indices = cite_link_splitter(link_data)
        link_data.train_idx = indices['train']
        link_data.val_idx = indices['valid']
        link_data.test_idx = indices['test']
        full_name = '/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/outdata'
        full_name = osp.join(full_name, dataset_name)
        generate_edge_level_prompt_files(full_name, link_data)
        print(f'Link level data for {dataset_name} has been generated')





