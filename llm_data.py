import os.path as osp
import torch
from graphllm.utils import generate_node_level_prompt_files, generate_edge_level_prompt_files, generate_multi_hop_x, generate_notestlink, generate_multi_hop_x_notestlink
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import index_to_mask
import os
## the following is the full set
# node_level_datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookchild', 'bookhis', 'elecomp', 'elephoto', 'sportsfit', 'products']

def load_local_data(dataset_name):
    path = "/egr/research-dselab/chenzh85/nips/MyOFA/cache_data_minilm"
    return torch.load(osp.join(path, dataset_name, 'processed', 'geometric_data_processed.pt'))[0]

def cite_link_splitter(data):
    edges = data.edge_index
    edge_perm = torch.randperm(len(edges[0]))
    train_offset = int(len(edge_perm) * 0.7)
    val_offset = int(len(edge_perm) * 0.8)
    edge_indices = {"train": edge_perm[:train_offset], "valid": edge_perm[train_offset:val_offset],
                    "test": edge_perm[val_offset:], }
    return edge_indices

def get_arxiv_mask():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_mask = index_to_mask(train_idx, size=dataset[0].num_nodes)
    valid_mask = index_to_mask(valid_idx, size=dataset[0].num_nodes)
    test_mask = index_to_mask(test_idx, size=dataset[0].num_nodes)
    return train_mask, valid_mask, test_mask


if __name__ == '__main__':
    train_mask, val_mask, test_mask = get_arxiv_mask()
    torch.save({'train': train_mask, 'valid': val_mask, 'test': test_mask}, 'arxiv_mask.pt')
    preprocessed_data_path = '/egr/research-dselab/chenzh85/nips/MyOFA/cache_data_minilm'
    raw_data_dir = "/egr/research-dselab/chenzh85/nips/MyOFA/data/single_graph/{}/categories.csv"

    # ## we first sanity check the simplest one
    # node_level_datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookchild', 'bookhis', 'elecomp', 'elephoto', 'sportsfit', 'products']
    node_level_datasets = ['dblp']
    # #node_level_datasets = []
    # link_level_datasets = []
    # link_level_datasets = ['cora', 'citeseer', 'pubmed']
    
    ## move raw label names
    for dataset_name in node_level_datasets:
        orig_file = raw_data_dir.format(dataset_name)
        full_name = '/egr/research-dselab/chenzh85/nips/MyOFA/outdata'         
        d_full_name = osp.join(full_name, dataset_name, 'categories.csv')
        path_name = osp.join(full_name, dataset_name)
        os.makedirs(path_name, exist_ok=True)
        os.system(f"cp {orig_file} {d_full_name}")

    for dataset_name in node_level_datasets:
        node_data = load_local_data(dataset_name)
        if dataset_name == 'arxiv':
            train_mask, val_mask, test_mask = get_arxiv_mask()
            node_data.train_mask = train_mask
            node_data.val_mask = val_mask
            node_data.test_mask = test_mask
        full_name = '/egr/research-dselab/chenzh85/nips/MyOFA/outdata'
        full_name = osp.join(full_name, dataset_name)
        generate_node_level_prompt_files(full_name, node_data)
        print(f'Node level data for {dataset_name} has been generated')

        generate_multi_hop_x(node_data, node_data.node_text_feat, full_name)
        print(f'Multi-hop data for {dataset_name} has been generated')

    # for dataset_name in link_level_datasets:
    #     link_data = load_local_data(dataset_name)
    #     link_dir = osp.join(preprocessed_data_path, dataset_name, 'processed')
    #     nolinkpath = generate_notestlink(link_data, link_dir)
    #     nolinkdata = torch.load(nolinkpath)
    #     generate_multi_hop_x_notestlink(nolinkdata, dataset_name, "./cache_data_minilm")
        # link_data = load_local_data(dataset_name)
        # indices = cite_link_splitter(link_data)
        # link_data.train_idx = indices['train']
        # link_data.val_idx = indices['valid']
        # link_data.test_idx = indices['test']
        # full_name = '/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/outdata'
        # full_name = osp.join(full_name, dataset_name)
        # generate_edge_level_prompt_files(full_name, link_data)
        # print(f'Link level data for {dataset_name} has been generated')





