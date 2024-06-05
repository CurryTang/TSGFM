import os.path as osp
import torch
from torch_geometric.utils import to_undirected






def load_one_tag_dataset(dataset = "cora", tag_data_path=""):
    AVAILABLE_DATASETS = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookhis', 
                          'bookchild', 'elephoto', 'elecomp', 'sportsfit', 'products', 'wikics', 
                          'cora-link', 'citeseer-link', 'pubmed-link', 'arxiv23-link', 'wikics-link', 
                          "arxiv-link", 'bookhis-link', 'bookchild-link', 'elephoto-link', 'elecomp-link', 'sportsfit-link', 'products-link']
    if dataset.endswith("-link"):
        dataset = dataset[:-5]
        link = True
    else:
        link = False
    lowers = [x.lower() for x in AVAILABLE_DATASETS]
    if dataset.lower() not in lowers:
        raise ValueError(f"Unknow dataset: {dataset}.")
    if tag_data_path == "":
        raise ValueError("tag_data_path is empty.")
    path = osp.join(tag_data_path, f"{dataset}/processed", "geometric_data_processed.pt")
    meta_data = osp.join(tag_data_path, f"{dataset}/processed", "data.pt")
    meta_data = torch.load(meta_data)
    if not link:
        meta_class_info = meta_data['e2e_node']['class_node_text_feat'][1]
    else:
        if meta_data.get('e2e_link'):
            meta_class_info = meta_data['e2e_link']['class_node_text_feat'][1]
        else:
            meta_class_info = ""
    if not osp.exists(path):
        raise ValueError(f"File not found: {path}")
    data = torch.load(path)[0]
    if meta_class_info != "":
        meta_class_emb = data.class_node_text_feat[meta_class_info]
    else:
        meta_class_emb = None
    feature = data.node_text_feat
    data.y = data.y.view(-1)
    # edge_index = data.edge_index
    # if dataset != 'arxiv23':
    data.edge_index = to_undirected(data.edge_index)
    data.x = feature
    data.meta_class_emb = meta_class_emb
    m_size = data.x.size(0)
    ## the following is for downstream tasks
    if not link:
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
    if not link:
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    return data