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

def get_label_names(dataframe):
    """Extracts label names in order of their first appearance."""
    unique_labels = dataframe['label'].unique()  # Get unique label values
    label_names = [dataframe.loc[dataframe['label'] == label, 'category'].iloc[0] for label in unique_labels]
    return label_names


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "children.csv")):
        csv_path = download_google_url("1mERB7AF31EGHbyfvQpKholgk1cTSCKBj", cur_path, "children.csv")
    else:
        csv_path = os.path.join(cur_path, "children.csv")
    if not os.path.exists(os.path.join(cur_path, "children.pt")):
        pt_path = download_google_url("1H_7Pmfg-8o3sLNflzWOHsgL5WG01i7B6", cur_path, "children.pt")
    else:
        pt_path = os.path.join(cur_path, "children.pt")
    label_desc = pd.read_csv(os.path.join(cur_path, "categories.csv"))    
    dgl_data = dgl.load_graphs(pt_path)[0][0]
    pd_data = pd.read_csv(csv_path)
    edges = dgl_data.edges()
    edge_index = torch.tensor([edges[0].tolist(), edges[1].tolist()], dtype=torch.long)
    pyg_data = pyg.data.Data(edge_index=edge_index, y=dgl_data.ndata['label'])
    dataset_size = pyg_data.y.shape[0]
    train_mask, val_mask, test_mask = generate_train_val_test_masks(dataset_size, 0.6, 0.2, 0.2)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    ## feat_node_texts
    feat_node_texts = pd_data['text'].tolist()
    feat_node_texts = ['feature node. paper title and abstract:' + t for t in feat_node_texts]
    ## class_node_texts
    class_node_texts = [
        "prompt node. literature category and description: "
        + line['name']
        + "."
        + line['description']
        for _, line in label_desc.iterrows()
    ]
    feat_edge_texts = ["feature edge. these two items are frequently co-purchased or co-viewed."] 
    noi_node_texts = ["prompt node. node classification of literature category, which country's history is it about?"]
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

