import os
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data.download import download_google_url
from torch_geometric.utils import coalesce, to_undirected
import numpy as np

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
    path = os.path.join(cur_path, "citeseer.pt")
    if not os.path.exists(path):
        download_google_url("1JRJHDiKFKiUpozGqkWhDY28E6F4v5n7l", cur_path, "citeseer.pt")
    data = torch.load(path)
    splitter = LabelPerClassSplit(num_labels_per_class=20, num_valid=500, num_test=-1)
    train_mask, val_mask, test_mask = splitter(data, data.x.shape[0])
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    text = data.raw_texts
    label_names = data.label_names
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(edge_index.size())
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = pyg.data.data.Data(**data_dict)
    new_data.edge_index = coalesce(to_undirected(new_data.edge_index))
    category_desc = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "categories.csv"), sep=","
    ).values
    ordered_desc = []
    for i, label in enumerate(label_names):
        true_ind = label == category_desc[:, 0]
        ordered_desc.append((label, category_desc[:, 1][true_ind]))
    clean_text = ["feature node. paper title and abstract: " + t for t in text]
    label_text = [
        "prompt node. literature category and description: "
        + desc[0]
        + "."
        + desc[1][0]
        for desc in ordered_desc
    ]
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
    noi_node_text = [
        "prompt node. node classification on the paper's category"
    ]
    prompt_edge_text = ["prompt edge", "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example"]
    return (
        [new_data],
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