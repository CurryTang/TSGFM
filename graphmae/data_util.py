
from collections import namedtuple, Counter
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import dgl
import random
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import index_to_mask, mask_to_index
from dgl.dataloading import DataLoader, ShaDowKHopSampler
from torch_geometric.utils import degree, one_hot
from dgl import RowFeatNormalizer
import torch_geometric


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["x"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["x"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_one_tag_dataset(dataset = "cora", tag_data_path=""):
    AVAILABLE_DATASETS = ['cora', 'citeseer', 'pubmed', 'arxiv', 'arxiv23', 'bookhis', 
                          'bookchild', 'elephoto', 'elecomp', 'sportsfit', 'products', 'wikics']
    lowers = [x.lower() for x in AVAILABLE_DATASETS]
    if dataset.lower() not in lowers:
        raise ValueError(f"Unknow dataset: {dataset}.")
    if tag_data_path == "":
        raise ValueError("tag_data_path is empty.")
    path = osp.join(tag_data_path, f"{dataset}/processed", "geometric_data_processed.pt")
    if not osp.exists(path):
        raise ValueError(f"File not found: {path}")
    data = torch.load(path)[0]
    feature = data.node_text_feat
    edge_index = data.edge_index
    labels = data.y
    labels = labels.reshape(-1)
    m_size = len(labels)
    ## the following is for downstream tasks
    if hasattr(data, "train_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    elif hasattr(data, 'train_masks'):
        train_mask = data.train_masks[0]
        val_mask = data.val_masks[0]
        test_mask = data.test_masks[0]
    elif hasattr(data, 'splits'):
        train_mask, val_mask, test_mask = [index_to_mask(data.splits['train'], size=m_size)], [index_to_mask(data.splits['valid'], size=m_size)], [index_to_mask(data.splits['test'], size = m_size)]
    return feature, edge_index, labels, train_mask, val_mask, test_mask
    



def load_pretrain_dataset(datasets, reptitions, args, is_link_pred = []):
    ## TODO: for link prediction task, remove the test edge index from the graph; passed from parameters is_link_pred
    graphs = []
    for dataset, rep in zip(datasets, reptitions):
        feature, edge_index, labels, _, _, _ = load_one_tag_dataset(dataset, args.tag_data_path)
        dgl_graph = dgl.graph(
            (edge_index[0], edge_index[1]),
            num_nodes=feature.shape[0],
        )
        dgl_graph.ndata["x"] = feature
        dgl_graph.ndata['y'] = labels
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        graphs.append({'g': dgl_graph, 'r': rep, 'name': dataset})
    return graphs


def set_partition_info(G, segment):
    segment.ndata['x'] = G.ndata['x'][segment.ndata['_ID']]
    segment.ndata['y'] = G.ndata['y'][segment.ndata['_ID']]
    return segment


def load_train_segments(graphs, args):
    ## reptitions are for continual learning, and fight catastrophic forgetting
    pre_train_loaders = []
    segment_data_map = {}
    total_segs = 0
    num_features = -1
    for i, gj in enumerate(graphs):
        ## split the train graph into partitions, halo node settings here
        g = gj['g']
        if i == 0:
            num_features = g.ndata['x'].shape[1]
        num_nodes = g.num_nodes()
        if num_nodes < args.partition_size:
            graph_parts = [g]
        else:
            parition_num = int(num_nodes / args.partition_size)
            graph_parts = dgl.metis_partition(g, parition_num, extra_cached_hops=1)
            graph_parts = graph_parts.values()
            graph_parts = [x for x in graph_parts]
            idx = np.random.permutation(parition_num)
            graph_parts = [graph_parts[i] for i in idx]
            graph_parts = [set_partition_info(g, x) for x in graph_parts]
        pre_train_loaders.extend(graph_parts)
        for _ in range(len(graph_parts)):
            segment_data_map[total_segs] = i
            total_segs += 1
    return pre_train_loaders, segment_data_map, num_features

def generate_one_hot_degree(edges, x, num_nodes, max_degree):
    idx, x = edges[0], x
    deg = degree(idx, num_nodes, dtype=torch.long)
    deg = one_hot(deg, num_classes=max_degree + 1)
    return deg



def load_downstream_dataset_pyg(args):
    datas = []
    for dataset in args.pre_train_datasets:
        feature, edge_index, labels, train_mask, val_mask, test_mask = load_one_tag_dataset(dataset, args.tag_data_path)
        edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index, None)
        pyg_graph =  torch_geometric.data.Data(x=feature, edge_index=edge_index, y=labels)
        pyg_graph.train_mask = train_mask
        pyg_graph.val_mask = val_mask
        pyg_graph.test_mask = test_mask
        datas.append(pyg_graph)
    return datas
        
        


def load_downstream_dataset(datasets, args):
    datas = []
    for dataset in datasets:
        feature, edge_index, labels, train_mask, val_mask, test_mask = load_one_tag_dataset(dataset, args.tag_data_path)
        dgl_graph = dgl.graph(
            (edge_index[0], edge_index[1]),
            num_nodes=feature.shape[0],
        )
        dgl_graph.ndata["x"] = feature
        if args.drop_feat:
            dgl_graph.ndata["x"] = generate_one_hot_degree(dgl_graph.edges(), feature, feature.shape[0], 256)
        if isinstance(train_mask, list):
            train_mask = train_mask[0]
            val_mask = val_mask[0]
            test_mask = test_mask[0]
        dgl_graph.ndata['train_mask'] = train_mask
        dgl_graph.ndata['val_mask'] = val_mask
        dgl_graph.ndata['test_mask'] = test_mask
        dgl_graph.ndata['y'] = labels
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        if args.drop_feat:
            norm = RowFeatNormalizer(subtract_min=True,node_feat_names=['x'])
            dgl_graph = norm(dgl_graph)
        datas.append(dgl_graph)
    return datas
        

def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)


def mask_to_split_idx(train_mask, val_mask, test_mask):
    split_idx = {}
    train_idx = mask_to_index(train_mask)
    val_idx = mask_to_index(val_mask)
    test_idx = mask_to_index(test_mask)
    split_idx["train"] = train_idx
    split_idx["valid"] = val_idx
    split_idx["test"] = test_idx
    return split_idx



## test here
if __name__ == '__main__':
    pass