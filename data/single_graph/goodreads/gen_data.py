import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url, download_url
import dgl
import numpy as np
import random

def select_unique_elements_1d(tensor, portion):
    """Selects a specified portion of elements from a 1D tensor without duplicates.

    Args:
        tensor (torch.Tensor): The input 1D tensor.
        portion (float): The portion of elements to select (between 0 and 1).

    Returns:
        torch.Tensor: A tensor containing the selected elements.
    """

    if portion <= 0 or portion > 1:
        raise ValueError("portion must be between 0 and 1")

    num_elements = int(portion * len(tensor))

    # Use a set to ensure uniqueness
    selected_indices = set(random.sample(range(len(tensor)), num_elements))

    selected_elements = tensor[list(selected_indices)]
    return selected_elements 

def subset_split(dataset, num_nodes):
    year = torch.randperm(num_nodes)
    train_offset = int(len(year) * 0.98)
    val_offset = int(len(year) * 0.01)
    test_offset = int(len(year) * 0.01)
    train_indices = year[:train_offset]
    val_indices = year[train_offset:train_offset + val_offset]
    test_indices = year[train_offset + val_offset:]
    train_indices = select_unique_elements_1d(train_indices, 0.03)
    val_indices = select_unique_elements_1d(val_indices, 0.25)
    test_indices = select_unique_elements_1d(test_indices, 0.5)
    return train_indices, val_indices, test_indices


def get_data(full = False):
    ## TODO: Change this, this is a link prediction dataset
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "goodreads.csv")):
        csv_path = download_google_url("1SGbbDeGYD51Yidk8EoNPZHAUsgjjHA8r", cur_path, "goodreads.csv")
    else:
        csv_path = os.path.join(cur_path, "goodreads.csv")
    if not os.path.exists(os.path.join(cur_path, "goodreads.pt")):
        pt_path = download_google_url("1WvDjZ2GB2nev3OXK5EARIInV0PCLsb5A", cur_path, "goodreads.pt")
    else:
        pt_path = os.path.join(cur_path, "goodreads.pt")  
    dgl_data = dgl.load_graphs(pt_path)[0][0]
    pd_data = pd.read_csv(csv_path)
    edges = dgl_data.edges()
    edge_index = torch.tensor([edges[0].tolist(), edges[1].tolist()], dtype=torch.long)
    pyg_data = pyg.data.Data(edge_index=edge_index, y=dgl_data.ndata['label'])
    dataset_size = pyg_data.y.shape[0]
    train_indices, val_indices, test_indices = subset_split(pyg_data)
    pyg_data.train_idx = train_indices
    pyg_data.val_idx = val_indices
    pyg_data.test_idx = test_indices
    ## feat_node_texts
    feat_node_texts = pd_data['text'].tolist()
    feat_node_texts = ['feature node. Book' + t for t in feat_node_texts]
    edge_label_text = [
        "prompt node. two books are similar",
        "prompt node. two books are very different"
    ]
    feat_edge_texts = ["feature edge. these two items are very similar."] 
    noi_node_texts = ["prompt node. node classification of literature category"]
    prompt_edge_texts = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    noi_node_edge_text = [
        "prompt node. link prediction on the papers that are cited together"
    ]
    return (
        [pyg_data],
        [
            feat_node_texts,
            feat_edge_texts,
            noi_node_texts + noi_node_edge_text,
            edge_label_text,
            prompt_edge_texts,
        ],
        {"e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(0, len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}
         }
    )

