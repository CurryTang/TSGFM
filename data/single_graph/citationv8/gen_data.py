import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url
import dgl
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

def subset_split(dataset):
    year = dataset.year 
    _, sorted_indices = torch.topk(year, k=len(year)) 
    train_offset = int(len(year) * 0.98)
    val_offset = int(len(year) * 0.01)
    test_offset = int(len(year) * 0.01)
    train_indices = sorted_indices[:train_offset]
    val_indices = sorted_indices[train_offset:train_offset + val_offset]
    test_indices = sorted_indices[train_offset + val_offset:]
    train_indices = select_unique_elements_1d(train_indices, 0.03)
    val_indices = select_unique_elements_1d(val_indices, 0.25)
    test_indices = select_unique_elements_1d(test_indices, 0.5)
    return train_indices, val_indices, test_indices


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "citationv8.csv")):
        csv_path = download_google_url("1qK9Jj-q2uXP69Gqw3jqLzkzEvdQv8VRQ", cur_path, "citationv8.csv")
    else:
        csv_path = os.path.join(cur_path, "citationv8.csv")
    if not os.path.exists(os.path.join(cur_path, "citationv8.pt")):
        pt_path = download_google_url("1xykV3HmvC9JWtHRO0-B4S4JqaWEfW8QM", cur_path, "citationv8.pt")
    else:
        pt_path = os.path.join(cur_path, "citationv8.pt")
    dgl_data = dgl.load_graphs(pt_path)[0][0]
    pd_data = pd.read_csv(csv_path)
    edges = dgl_data.edges()
    edge_index = torch.tensor([edges[0].tolist(), edges[1].tolist()], dtype=torch.long)
    new_data = pyg.data.Data(edge_index=edge_index, node_year = dgl_data.ndata["year"])
    train_indices, val_indices, test_indices = subset_split(new_data)
    new_data.train_idx = train_indices
    new_data.val_idx = val_indices
    new_data.test_idx = test_indices
    text = pd_data["text"].tolist()
    clean_text = ["feature node. paper title and abstract: " + t for t in text]
    edge_label_text = [
        "prompt node. two papers do not have co-citation",
        "prompt node. two papers have co-citation"
    ]
    edge_text = [
        "feature edge. connected papers are cited together by other papers."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the papers that are cited together"
    ]
    prompt_edge_text = ["prompt edge", "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example"]
    noi_node_text = [
        "prompt node. node classification on the paper's category"
    ]
    return (
        [new_data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            edge_label_text,
            prompt_edge_text,
        ],
        {"e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(0, len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}
         }
    )
