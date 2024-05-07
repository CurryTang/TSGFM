import os
import pandas as pd
import torch
from data.ofa_data import OFAPygDataset
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np

def get_node_feature(path):
    # Node feature process
    nodeidx2paperid = pd.read_csv(os.path.join(path, "nodeidx2paperid.csv.gz"), index_col="node idx")
    titleabs_url = ("https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv")
    titleabs = pd.read_csv(titleabs_url, sep="\t", names=["paper id", "title", "abstract"], index_col="paper id", )

    titleabs = nodeidx2paperid.join(titleabs, on="paper id")
    text = ("feature node. paper title and abstract: " + titleabs["title"] + ". " + titleabs["abstract"])
    node_text_lst = text.values

    return node_text_lst


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


def get_logic_feature(path):
    arxiv_categ_vals = get_pd_feature(path)
    or_labeled_text = []
    not_and_labeled_text = []
    for i in range(len(arxiv_categ_vals)):
        for j in range(len(arxiv_categ_vals)):
            c1 = arxiv_categ_vals.iloc[i]
            c2 = arxiv_categ_vals.iloc[j]
            txt = "prompt node. literature category and description: not " + c1["name"] + ". " + c1[
                "description"] + " and not " + c2["name"] + ". " + c2["description"]
            not_and_labeled_text.append(txt)
            txt = "prompt node. literature category and description: either " + c1["name"] + ". " + c1[
                "description"] + " or " + c2["name"] + ". " + c2["description"]
            or_labeled_text.append(txt)
    return or_labeled_text + not_and_labeled_text

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def get_data(dset):
    pyg_data = PygNodePropPredDataset(name="ogbn-arxiv", root=dset.data_dir)
    years = pyg_data[0].node_year.reshape(-1)
    label = even_quantile_labels(
        years, 5, verbose=False)
    pyg_data.data.y = torch.as_tensor(label)
    cur_path = os.path.dirname(__file__)
    splits = pyg_data.get_idx_split()
    pyg_data.data.splits = splits
    feat_node_texts = get_node_feature(cur_path).tolist()
    # class_node_texts = get_label_feature(cur_path).tolist()
    label_desc = pd.read_csv(os.path.join(cur_path, "categories.csv"))    
    class_node_texts = [
        "prompt node. category and description: "
        + line['name']
        + "."
        + line['description']
        for _, line in label_desc.iterrows()
    ]
    # logic_node_texts = get_logic_feature(cur_path)
    feat_edge_texts = ["feature edge. citation"]
    noi_node_texts = ["prompt node. node classification of literature category"]
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
    return ([pyg_data.data], [feat_node_texts, feat_edge_texts, noi_node_texts, class_node_texts,
        prompt_edge_texts, ], prompt_text_map,)