from data.ofa_data import OFAPygDataset
import torch
import os.path as osp
import os
import pandas as pd
import numpy as np

def gen_graph(subg_path, categories_csv):
    print("Generate the raw files for MAG240M")
    data_path = subg_path
    ## text don't do anything
    ## only have one split
    subg_mapping = osp.join(data_path, "mag240m_mapping.pt")
    data_file = (data_path, subg_mapping)
    prompt_edge_text = ["prompt edge.", "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example", ]
    
    prompt_text = ["prompt node. node classification on the category of this paper",
        "prompt node. few shot task node for node classification that decides which class the query molecule belongs to ",
        "the class of support papers.", ]

    label_desc = pd.read_csv(categories_csv)    

    labels_features = [
        "prompt node. literature category and description: "
        + line['name']
        + "."
        + line['description']
        for _, line in label_desc.iterrows()
    ]

    prompt_text_map = {
        "e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                    "class_node_text_feat": ["class_node_text_feat",
                                            torch.arange(len(labels_features))],
                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
        "lr_node": {"noi_node_text_feat": ["noi_node_text_feat", [1]],
                    "class_node_text_feat": ["class_node_text_feat",
                                            torch.arange(len(labels_features))],
                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0,1,2]]}}
                    

    return (
        data_file, 
        [[], [], labels_features, prompt_text, prompt_edge_text],
        prompt_text_map,
    )

class SegmentDataset(OFAPygDataset):
    def gen_data(self):
        pyg_data_path, texts, split = gen_graph("subg", "categories.csv")
        self.pyg_data_path = pyg_data_path
        return pyg_data_path, texts, split

    def add_text_emb(self, data_list, text_emb):
        """
        Since the majority of node/edge text embeddings are repeated, we only store unique
        ones, and keep the indices.
        """
        data, slices = self.collate(data_list)
        # data.node_embs = np.
        # data.node_embs = text_emb[0]
        # data.edge_embs = text_emb[1]
        feats = np.load("node_feat_link", mmap_mode='r')
        data.node_embs = feats
        data.class_node_text_feat = text_emb[2]
        data.prompt_edge_text_feat = text_emb[3]
        data.noi_node_text_feat = text_emb[4]
        return data, slices

    def get(self, index):
        data = super().get(index)
        node_feat = torch.tensor(self.node_embs[data.x])
        # edge_feat = self.edge_embs[data.xe]
        data.node_text_feat = node_feat
        # data.edge_text_feat = edge_feat
        data.y = data.y.view(1, -1)
        return data

    def get_idx_split(self):
        return self.side_data[0]

    def get_task_map(self):
        return self.side_data[1]

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_node":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_node":
            return {"f2n": [1, 0]}