import os
import pandas as pd
import torch
from ogb.utils.torch_util import replace_numpy_with_torchtensor
import json
import logging
import os.path as osp
import shutil
import gdown
import numpy as np
import torch.distributed as dist
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from transformers import AutoTokenizer
from torch_geometric.utils.mask import index_to_mask
from contextlib import contextmanager
from ogb.utils.url import decide_download, download_url, extract_zip
from tqdm import tqdm

## this dataset is too large to be trained in the OFA way


logger = logging.getLogger(__name__)

@contextmanager
def dist_barrier_context():
    rank = int(os.getenv("RANK", -1))
    if rank not in [0, -1]:
        dist.barrier()
    yield
    if rank == 0:
        dist.barrier()


class OgbWithText(InMemoryDataset):
    def __init__(
        self,
        name,
        meta_info,
        root="data",
        transform=None,
        pre_transform=None,
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        tokenize=True,
    ):
        self.name = name  ## original name, e.g., ogbn-proteins
        self.meta_info = meta_info
        self.dir_name = "_".join(self.name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.should_tokenize = tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True) if tokenize else None
        # check if the dataset is already processed with the same tokenizer
        rank = int(os.getenv("RANK", -1))
        with dist_barrier_context():
            super(OgbWithText, self).__init__(self.root, transform, pre_transform)
        if rank in [0, -1] and tokenize:
            self.save_metainfo()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # add input_ids and attention_mask
        if self.should_tokenize:
            if not osp.exists(self.tokenized_path) and rank <= 0:
                _ = self.mapping_and_tokenizing()
            dist.barrier()
            self._data.input_ids, self._data.attention_mask = self.load_cached_tokens()

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def _mapping_and_tokenizing(self):
        raise NotImplementedError

    @property
    def tokenized_path(self):
        tokenizer_name = "_".join(self.tokenizer.name_or_path.split("/"))
        tokenized_path = osp.join(self.root, "processed", f"{tokenizer_name}.pt")
        return tokenized_path

    def load_cached_tokens(self):
        if osp.exists(self.tokenized_path):
            logger.info("using cached tokenized data in {}".format(self.tokenized_path))
            text_encoding = torch.load(self.tokenized_path)
            return text_encoding["input_ids"], text_encoding["attention_mask"]

    def mapping_and_tokenizing(self):
        input_ids, attention_mask = self._mapping_and_tokenizing()
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, self.tokenized_path)
        logger.info("save the tokenized data to {}".format(self.tokenized_path))
        return input_ids, attention_mask

    def save_metainfo(self):
        w_path = osp.join(self.root, "processed/meta_info.json")
        with open(w_path, "w") as outfile:
            json.dump(self.meta_info, outfile)

    def load_metainfo(self):
        r_path = osp.join(self.root, "processed/meta_info.json")
        if not osp.exists(r_path):
            return None
        return json.loads(open(r_path).read())

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class OgblCitation2WithText(OgbWithText):
    def __init__(
        self,
        root="data",
        transform=None,
        pre_transform=None,
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        tokenize=True,
    ):
        name = "ogbl-citation2-text"  ## original name, e.g., ogbl-ppa

        meta_info = {
            "download_name": "citation-v2",
            "task_type": "link prediction",
            "eval_metric": "mrr",
            "add_inverse_edge": False,
            "version": 1,
            "has_node_attr": True,
            "has_edge_attr": False,
            "split": "time",
            "additional_node_files": ["node_year"],
            "additional_edge_files": [],
            "is_hetero": False,
            "binary": False,
            "graph_url": "http://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip",
            "text_url": "https://drive.google.com/u/0/uc?id=19_hkbBUDFZTvQrM0oMbftuXhgz5LbIZY&export=download",
            "tokenizer": tokenizer,
        }
        super(OgblCitation2WithText, self).__init__(
            name, meta_info, root, transform, pre_transform, tokenizer, tokenize
        )

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)

        split_idx = {"train": None, "valid": None, "test": None}
        for key, item in split_idx.items():
            split_idx[key] = replace_numpy_with_torchtensor(torch.load(osp.join(path, f"{key}.pt")))

        # subset with subset_node_idx and relable nodes
        subset_node_idx = self._get_subset_node_idx()
        num_nodes, edge_index = 0, {}
        for key in split_idx.keys():
            edge_index[key] = torch.stack([split_idx[key]["source_node"], split_idx[key]["target_node"]], dim=0)
            if key in ["valid", "test"]:
                edge_index[key] = torch.cat([edge_index[key], split_idx[key]["target_node_neg"].t()], dim=0)
            num_nodes = max(num_nodes, int(edge_index[key].max()) + 1)

        def subset_and_relabeling(edge_index, subset, num_nodes):
            node_mask = index_to_mask(subset, size=num_nodes)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_index = edge_index[:, edge_mask]
            # relabel
            node_idx = torch.zeros(num_nodes, dtype=torch.long)
            node_idx[node_mask] = torch.arange(node_mask.sum())
            edge_index = node_idx[edge_index]
            # BUG: some out-of-subset edges may appear, relabel them to 0
            return edge_index

        for key in split_idx.keys():
            edge_index[key] = subset_and_relabeling(edge_index[key], subset_node_idx, num_nodes)
            split_idx[key]["source_node"], split_idx[key]["target_node"] = edge_index[key][0], edge_index[key][1]
            if key in ["valid", "test"]:
                split_idx[key]["target_node_neg"] = edge_index[key][2:].t()

        return split_idx

    @property
    def raw_file_names(self):
        if self.meta_info["binary"]:
            if self.meta_info["is_hetero"]:
                return ["edge_index_dict.npz"]
            else:
                return ["data.npz"]
        else:
            if self.meta_info["is_hetero"]:
                return ["num-node-dict.csv.gz", "triplet-type-list.csv.gz"]
            else:
                file_names = ["edge"]
                if self.meta_info["has_node_attr"] == "True":
                    file_names.append("node-feat")
                if self.meta_info["has_edge_attr"] == "True":
                    file_names.append("edge-feat")
                return [file_name + ".csv.gz" for file_name in file_names]

    def download(self):
        graph_url = self.meta_info["graph_url"]
        if decide_download(graph_url):
            path = download_url(graph_url, self.original_root)
            extract_zip(path, self.original_root)
            # download text data from google drive
            output = osp.join(self.original_root, self.meta_info["download_name"], "raw", "idx_title_abs.csv.gz")
            if osp.exists(output) and osp.getsize(output) > 0:
                logger.info(f"Using existing file {output}")
            else:
                gdown.download(url=self.meta_info["text_url"], output=output, quiet=False, fuzzy=False)
            # cleanup
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.meta_info["download_name"]), self.root)
        else:
            logger.warning("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        data = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=add_inverse_edge,
            additional_node_files=self.meta_info["additional_node_files"],
            additional_edge_files=self.meta_info["additional_edge_files"],
            binary=self.meta_info["binary"],
        )[0]

        data = data if self.pre_transform is None else self.pre_transform(data)
        subset_node_idx = self._get_subset_node_idx()
        data = data.subgraph(subset_node_idx)

        print("Saving...")
        torch.save(self.collate([data]), self.processed_paths[0])

    def _mapping_and_tokenizing(self):
        df = pd.read_csv(osp.join(self.raw_dir, "idx_title_abs.csv.gz"))
        df.sort_values(by="node idx", inplace=True)
        df["abstitle"] = "title: " + df["title"] + "; " + "abstract: " + df["abstract"]
        input_ids, attention_mask, truncated_size = [], [], 10000
        text_list = df["abstitle"].values.tolist()
        print("Tokenizing...")
        for i in tqdm(range(0, len(df), truncated_size)):
            j = min(len(text_list), i + truncated_size)
            _encodings = self.tokenizer(text_list[i:j], padding=True, truncation=True, return_tensors="pt")
            input_ids.append(_encodings.input_ids)
            attention_mask.append(_encodings.attention_mask)
        input_ids, attention_mask = torch.cat(input_ids, dim=0), torch.cat(attention_mask, dim=0)
        return input_ids, attention_mask

    def _get_subset_node_idx(self):
        df = pd.read_csv(osp.join(self.raw_dir, "idx_title_abs.csv.gz"))
        df.astype({"node idx": np.int64})
        node_idx = torch.tensor(df["node idx"].values.tolist(), dtype=torch.long)
        return node_idx