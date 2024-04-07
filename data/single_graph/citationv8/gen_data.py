import os
import pandas as pd
import torch
import torch_geometric as pyg
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url
import dgl



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
