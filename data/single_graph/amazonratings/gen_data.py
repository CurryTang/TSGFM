import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url, download_url
import dgl
import numpy as np

def generate_train_val_test_masks(dataset_size, train_ratio, validation_ratio, test_ratio):
    """Generates training, validation, and testing masks as PyTorch tensors.

    Args:
        dataset_size: The total number of data points in the dataset.
        train_ratio: The proportion of data to be used for training.
        validation_ratio: The proportion of data to be used for validation.
        test_ratio: The proportion of data to be used for testing.

    Returns:
        tuple: A tuple containing the training mask, validation mask, and testing mask.
    """

    if train_ratio + validation_ratio + test_ratio != 1:
        raise ValueError("Ratios must sum up to 1")

    num_train = int(dataset_size * train_ratio)
    num_val = int(dataset_size * validation_ratio)
    num_test = dataset_size - num_train - num_val

    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_mask = torch.zeros(dataset_size, dtype=torch.bool)
    train_mask[indices[:num_train]] = True

    val_mask = torch.zeros(dataset_size, dtype=torch.bool)
    val_mask[indices[num_train:num_train + num_val]] = True

    test_mask = torch.zeros(dataset_size, dtype=torch.bool)
    test_mask[indices[num_train + num_val:]] = True

    return train_mask, val_mask, test_mask

def get_data(dset):
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "ratings.pt")):
        pt_path = download_google_url("1-LyPmaOxSTUPD2J0a_2NUk7-PIQi177_", cur_path, "ratings.pt")
    else:
        pt_path = os.path.join(cur_path, "ratings.pt")
    label_desc = [
        "5 score awesome ratings, users are extremely satisfied with the products.",
        "4.5 score good ratings, users are satisfied with the products, but there's space to be even better.",
        "4 score good ratings, users like the products but there's still much space to be better.",
        "3.5 score average ratings, users are neutral about the products.",
        "0-3 score bad ratings, users think the products are bad."
    ]    
    pyg_data = torch.load(pt_path)
    dataset_size = pyg_data.y.shape[0]
    train_mask, val_mask, test_mask = generate_train_val_test_masks(dataset_size, 0.5, 0.25, 0.25)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    ## feat_node_texts
    feat_node_texts = pyg_data.raw_texts
    feat_node_texts = ['feature node. Name of the products' + t for t in feat_node_texts]
    ## class_node_texts
    class_node_texts = [
        "prompt node. Score of the products: " + x for x in label_desc 
    ]
    feat_edge_texts = ["feature edge. these two items are frequently co-purchased or co-viewed."] 
    noi_node_texts = ["prompt node. node classification of the score of the products?"]
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

