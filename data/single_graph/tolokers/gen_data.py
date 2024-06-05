import os
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_undirected
import dgl
import numpy as np
import requests

def process_files(cur_path):
    edges = pd.read_csv(os.path.join(cur_path, "edges.tsv"), sep="\t")
    nodes = pd.read_csv(os.path.join(cur_path, "nodes.tsv"), sep="\t")
    splits_train = pd.read_csv(os.path.join(cur_path, "splits_train.tsv"), sep="\t")
    splits_val = pd.read_csv(os.path.join(cur_path, "splits_val.tsv"), sep="\t")
    splits_test = pd.read_csv(os.path.join(cur_path, "splits_test.tsv"), sep="\t")

    train_mask = torch.from_numpy(splits_train['1'].to_numpy()).bool()
    val_mask = torch.from_numpy(splits_val['1'].to_numpy()).bool()
    test_mask = torch.from_numpy(splits_test['1'].to_numpy()).bool()

    source = []
    target = []
    attrs = []

    for row in edges.iterrows():
        sourceid = row[1]["source"]
        targetid = row[1]["target"]
        
        source.append(sourceid)
        target.append(targetid)
    
    edge_index = torch.tensor([source, target], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    labels = torch.from_numpy(nodes['banned'].to_numpy()).long()

    for row in nodes.iterrows():
        approved_rate=row[1]["approved_rate"]
        skipped_rate=row[1]["skipped_rate"]	
        expired_rate=row[1]["expired_rate"]	
        rejected_rate=row[1]["rejected_rate"]	
        education=row[1]["education"]	
        english_profile=row[1]["english_profile"]
        english_tested=row[1]["english_tested"]

        english_profile = "yes" if english_profile == 1 else 0
        english_tested = "yes" if english_tested == 1 else 0

        attr = "This user has an approved rate of " + str(approved_rate) + " and a skipped rate of " + str(skipped_rate) + " and an expired rate of " + str(expired_rate) + " and a rejected rate of " + str(rejected_rate) + " and an education level of " + str(education) + " and an english profile of " + str(english_profile) + " and an english tested of " + str(english_tested)

        attrs.append(attr)
    return attrs, edge_index, labels, train_mask, val_mask, test_mask

def get_data(dset):
    cur_path = os.path.dirname(__file__)
    edges_tsv = "https://raw.githubusercontent.com/Toloka/TolokerGraph/main/edges.tsv"
    nodes_tsv = "https://raw.githubusercontent.com/Toloka/TolokerGraph/main/nodes.tsv"
    splits_train_tsv = "https://raw.githubusercontent.com/Toloka/TolokerGraph/main/splits_train.tsv"
    splits_val_tsv = "https://raw.githubusercontent.com/Toloka/TolokerGraph/main/splits_val.tsv"
    splits_test_tsv = "https://raw.githubusercontent.com/Toloka/TolokerGraph/main/splits_test.tsv"
    raw_files = [edges_tsv, nodes_tsv, splits_train_tsv, splits_val_tsv, splits_test_tsv]
    for raw_file in raw_files:
        if not os.path.exists(os.path.join(cur_path, raw_file.split("/")[-1])):
            with open(os.path.join(cur_path, raw_file.split("/")[-1]), 'wb') as f:
                f.write(requests.get(raw_file).content)
    label_desc = [
        "class 0: this annotator behaves normally and is not banned in this project",
        "class 1: this annotator does something bad and is banned in this project"
    ]
    feat_node_texts, edge_index, labels, train_mask, val_mask, test_mask = process_files(cur_path)    
    pyg_data = pyg.data.Data(x=None, edge_index=edge_index, y=labels)
    dataset_size = pyg_data.y.shape[0]
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    ## feat_node_texts
    feat_node_texts = ['feature node. Attribute of user' + t for t in feat_node_texts]
    ## class_node_texts
    class_node_texts = [
        "prompt node. Whether this user has been banned " + x for x in label_desc 
    ]
    feat_edge_texts = ["feature edge. these two users share the same identifier."] 
    noi_node_texts = ["prompt node. node classification of the category of user"]
    prompt_edge_texts = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             torch.arange(len(class_node_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    return ([pyg_data], [feat_node_texts, feat_edge_texts, noi_node_texts, class_node_texts,
        prompt_edge_texts, ], prompt_text_map,)



if __name__ == "__main__":
    get_data("tolokers")