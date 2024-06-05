import os
import pandas as pd
import torch
import numpy as np
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data.download import download_google_url, download_url
import csv
import codecs


def get_taxonomy(path):
    # read categories and description file
    f = open(os.path.join(path, "arxiv_CS_categories.txt"), "r").readlines()

    state = 0
    result = {"id": [], "name": [], "description": []}

    for line in f:
        if state == 0:
            assert line.strip().startswith("cs.")
            category = ("arxiv " + " ".join(line.strip().split(" ")[0].split(".")).lower())  # e. g. cs lo
            name = line.strip()[7:-1]  # e. g. Logic in CS
            result["id"].append(category)
            result["name"].append(name)
            state = 1
            continue
        elif state == 1:
            description = line.strip()
            result["description"].append(description)
            state = 2
            continue
        elif state == 2:
            state = 0
            continue

    arxiv_cs_taxonomy = pd.DataFrame(result)

    return arxiv_cs_taxonomy


def get_pd_feature(path):
    arxiv_cs_taxonomy = get_taxonomy(path)
    mapping_file = os.path.join(path, "labelidx2arxivcategeory.csv.gz")
    arxiv_categ_vals = pd.merge(pd.read_csv(mapping_file), arxiv_cs_taxonomy, left_on="arxiv category", right_on="id", )
    return arxiv_categ_vals


def get_label_feature(path):
    arxiv_categ_vals = get_pd_feature(path)
    text = ("prompt node. literature category and description: " + arxiv_categ_vals["name"] + ". " + arxiv_categ_vals[
        "description"])
    label_text_lst = text.values

    return label_text_lst



def clean_csv_utf8_inline(filename):
    """
    Removes non-UTF-8 characters from a CSV file, modifying it directly.

    Args:
        filename: The path to the CSV file to clean.
    """

    temp_filename = filename + ".tmp"

    with codecs.open(filename, "r", encoding="utf-8", errors="ignore") as infile, \
         open(temp_filename, "w", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            cleaned_row = [cell.encode("utf-8", errors="replace").decode("utf-8") for cell in row]
            writer.writerow(cleaned_row)

    # Replace original file with the cleaned version
    infile.close()
    outfile.close()
    import os
    os.remove(filename)
    os.rename(temp_filename, filename)


def get_data(dset):
    cur_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cur_path, "arxiv23.csv")):
        csv_path = download_google_url("1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv", cur_path, "arxiv23.csv")
        clean_csv_utf8_inline(csv_path)
    else:
        csv_path = os.path.join(cur_path, "arxiv23.csv")
    if not os.path.exists(os.path.join(cur_path, "arxiv23.pt")):
        pyg_path = download_url("https://github.com/XiaoxinHe/TAPE/raw/main/dataset/arxiv_2023/graph.pt", folder=cur_path, filename="arxiv23.pt")
    else:
        pyg_path = os.path.join(cur_path, "arxiv23.pt")
    pyg_data = torch.load(pyg_path)
    pd_data = pd.read_csv(csv_path)
    feat_node_texts = (pd_data['title'] + ':' + pd_data['abstract']).to_list()
    class_node_texts = get_label_feature(cur_path).tolist()
    feat_edge_texts = ["feature edge. citation"]
    noi_node_texts = ["prompt node. node classification of literature category"]
    prompt_edge_texts = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
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
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(class_node_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                    "e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(len(class_node_texts), len(class_node_texts) + len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                   "class_node_text_feat": ["class_node_text_feat",
                                                            torch.arange(len(class_node_texts))],
                                   "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}
    return ([pyg_data], [feat_node_texts, feat_edge_texts, noi_node_texts + noi_node_edge_text, class_node_texts + edge_label_text,
        prompt_edge_texts, ], prompt_text_map,)
