import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url, download_url
import dgl
import numpy as np


import pandas as pd
from collections import Counter

def generate_masks_by_year(year_list, train_ratio, valid_ratio, test_ratio):
    """
    Generates train, validation, and test masks based on provided years and ratios.

    Args:
        year_list (list): A list of integers representing years.
        train_ratio (float): The proportion of data to use for the training set (between 0 and 1).
        valid_ratio (float): The proportion of data to use for the validation set (between 0 and 1).
        test_ratio (float): The proportion of data to use for the test set (between 0 and 1).

    Returns:
        tuple: A tuple containing PyTorch tensors representing the train, validation, and test masks.
    """

    # Input validation
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("Ratios must sum up to 1")

    # Sort years in ascending order
    years = np.array(year_list)
    sorted_indices = years.argsort()

    # Calculate indices for splits
    num_samples = len(years)
    train_end_idx = int(num_samples * train_ratio)
    valid_end_idx = train_end_idx + int(num_samples * valid_ratio)

    # Create masks
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    valid_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    train_mask[sorted_indices[:train_end_idx]] = True
    valid_mask[sorted_indices[train_end_idx:valid_end_idx]] = True
    test_mask[sorted_indices[valid_end_idx:]] = True

    return train_mask, valid_mask, test_mask 



def get_label_names(dataframe):
    """Extracts label names in order of their first appearance."""
    unique_labels = dataframe['label'].unique()  # Get unique label values
    label_names = [dataframe.loc[dataframe['label'] == label, 'category'].iloc[0] for label in unique_labels]
    return label_names


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "elephoto.csv")):
        csv_path = download_google_url("1HhR-XIga6x00r1eWqBl_y5oHhIKnBFtw", cur_path, "elephoto.csv")
    else:
        csv_path = os.path.join(cur_path, "elephoto.csv")
    if not os.path.exists(os.path.join(cur_path, "elephoto.pt")):
        pt_path = download_google_url("1PeXGzCrITyg3ItuFZL45VqoNtget4cdL", cur_path, "elephoto.pt")
    else:
        pt_path = os.path.join(cur_path, "elephoto.pt")
    label_desc = pd.read_csv('./categories.csv')    
    dgl_data = dgl.load_graphs(pt_path)[0][0]
    pd_data = pd.read_csv(csv_path)
    edges = dgl_data.edges()
    edge_index = torch.tensor([edges[0].tolist(), edges[1].tolist()], dtype=torch.long)
    pyg_data = pyg.data.Data(edge_index=edge_index, y=dgl_data.ndata['label'])
    train_mask, val_mask, test_mask = generate_masks_by_year(dgl_data.ndata['year'].tolist(), 0.6, 0.2, 0.2)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    ## feat_node_texts
    feat_node_texts = pd_data['text'].tolist()
    feat_node_texts = ['feature node. Review: ' + t for t in feat_node_texts]
    ## class_node_texts
    class_node_texts = [
        "prompt node. Photo product category and description: "
        + line['name']
        + "."
        + line['description']
        for _, line in label_desc.iterrows()
    ]
    feat_edge_texts = ["feature edge. these two items are frequently co-purchased or co-viewed."] 
    noi_node_texts = ["prompt node. node classification of product category"]
    prompt_edge_texts = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(class_node_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                   "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(class_node_texts))],
                                   "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}
    return ([pyg_data], [feat_node_texts, feat_edge_texts, noi_node_texts, class_node_texts,
        prompt_edge_texts, ], prompt_text_map,)