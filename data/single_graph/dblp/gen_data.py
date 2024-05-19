import os
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data.download import download_google_url
from torch_geometric.utils import coalesce, to_undirected
import numpy as np
from collections import defaultdict


def construct_pyg_graph():
    cur_path = os.path.dirname(__file__)
    pa_rel = os.path.join(cur_path, "paper_author.txt")
    paper = os.path.join(cur_path, "paper.txt")
    paperid2id = {}
    id2paperid = {}
    raw_texts = []
    with open(paper, "r") as f:
        for i, line in enumerate(f):
            cline = line.strip().split("\t")
            paperid = int(cline[0])
            papertitle = cline[1]
            raw_texts.append(papertitle)
            paperid2id[paperid] = i
            id2paperid[i] = paperid
    
    dense_adj = torch.zeros((len(paperid2id), len(paperid2id)))
    same_author_dict = defaultdict(list)
    with open(pa_rel, "r") as f:
        for line in f:
            cline = line.strip().split("\t")
            paperid = int(cline[0])
            authorid = int(cline[1])
            same_author_dict[authorid].append(paperid2id[paperid])
    
    for author, papers in same_author_dict.items():
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                dense_adj[papers[i], papers[j]] = 1
                dense_adj[papers[j], papers[i]] = 1



    edge_index = to_undirected(coalesce(dense_adj.nonzero().t().contiguous()))


    conf_label_map = {
        '36': 2,
        '597': 3,
        '755': 2,
        '1194': 3,
        '1201': 2,
        '1234': 0,
        '1798': 0,
        '1801': 1,
        '1902': 2,
        '2180': 2,
        '2504': 1,
        '2934': 1,
        '3011': 1,
        '3027': 0,
        '3230': 1,
        '3318': 3,
        '3329': 0,
        '3594': 0,
        '3771': 3,
        '4096': 3
        }
    
    paper_conf = os.path.join(cur_path, "paper_conf.txt")

    labels = [-1 for _ in range(len(paperid2id))]
    with open(paper_conf, "r") as f:
        for line in f:
            cline = line.strip().split("\t")
            paperid = int(cline[0])
            confid = cline[1]
            assert paperid in paperid2id
            assert confid in conf_label_map
            labels[paperid2id[paperid]] = conf_label_map[confid]
    
    labels = torch.tensor(labels)
    return edge_index, labels, raw_texts

            





class LabelPerClassSplit(object):
    """
    Class for splitting data into training, validation, and test sets based on labels.

    This class provides a callable object for splitting data into training, validation, and test sets.
    The splitting is done based on the labels of the data, with a specified number of labels per class for the training set.
    """
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
            num_test: int = 1000,
            inside_old_mask: bool = False
    ):
        """
        Constructor method for the LabelPerClassSplit class.

        Initializes a new instance of the LabelPerClassSplit class with the provided parameters.

        Parameters:
        num_labels_per_class (int, optional): The number of labels per class for the training set. Defaults to 20.
        num_valid (int, optional): The number of validation data points. Defaults to 500.
        num_test (int, optional): The number of test data points. If -1, all remaining data points after training and validation are used for testing. Defaults to -1.
        inside_old_mask (bool, optional): Whether to consider only data points inside the old mask for splitting. Defaults to False.

        Returns:
        None
        """
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid
        self.num_test = num_test
        self.inside_old_mask = inside_old_mask

    def __call__(self, data, total_num):
        """
        Callable method for the LabelPerClassSplit class.

        This method splits the data into training, validation, and test sets based on the labels of the data.

        Parameters:
        data: The data to be split.
        total_num (int): The total number of data points.

        Returns:
        tuple: A tuple containing the masks for the training, validation, and test sets.
        """
        new_train_mask = torch.zeros(total_num, dtype=torch.bool)
        new_val_mask = torch.zeros(total_num, dtype=torch.bool)
        new_test_mask = torch.zeros(total_num, dtype=torch.bool)

        perm = torch.randperm(total_num)
        train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int32)

        for i in range(perm.numel()):
            label = data.y[perm[i]]
            if train_cnt[label] < self.num_labels_per_class:
                train_cnt[label] += 1
                new_train_mask[perm[i]] = 1
            elif new_val_mask.sum() < self.num_valid:
                new_val_mask[perm[i]] = 1
            else:
                if new_test_mask.sum() < self.num_test:
                    new_test_mask[perm[i]] = 1
                else:
                    break
        
        if self.num_test == -1:
            new_test_mask = ~new_train_mask & ~new_val_mask

        return new_train_mask, new_val_mask, new_test_mask



def get_data(dset):
    cur_path = os.path.dirname(__file__)
    edge_index, labels, raw_texts = construct_pyg_graph()
    data = pyg.data.Data(
        x=None, edge_index=edge_index, y=labels)
    splitter = LabelPerClassSplit(num_labels_per_class=20, num_valid=500, num_test=-1)
    train_mask, val_mask, test_mask = splitter(data, len(labels))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    text = raw_texts
    label_names = ['Database', 'Data Mining', 'AI', 'Information Retrieval']
    category_desc = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "categories.csv"), sep=","
    ).values
    ordered_desc = []
    for i, label in enumerate(label_names):
        true_ind = label == category_desc[:, 0]
        ordered_desc.append((label, category_desc[:, 1][true_ind]))
    clean_text = ["feature node. paper title: " + t for t in text]
    label_text = [
        "prompt node. literature category and description: "
        + desc[0]
        + "."
        + desc[1][0]
        for desc in ordered_desc
    ]
    edge_label_text = [
        "prompt node. two papers are written by the same author",
        "prompt node. two papers are written by different authors"
    ]
    edge_text = [
        "feature edge. connected papers are written by the same author",
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the papers that are written by the same authors"
    ]
    noi_node_text = [
        "prompt node. node classification on the paper's category"
    ]
    prompt_edge_text = ["prompt edge", "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example"]
    return (
        [data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            label_text + edge_label_text,
            prompt_edge_text,
        ],
        {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                      "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         "e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                      "class_node_text_feat": ["class_node_text_feat",
                                               torch.arange(len(label_text), len(label_text) + len(edge_label_text))],
                      "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
         "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                     "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                     "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}
         }
    )