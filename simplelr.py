from graphmae.data_util import load_one_tag_dataset
from graphmae.utils import build_args
from torch_geometric.utils import to_undirected, remove_self_loops, degree
from graphllm.utils import MP
import torch.nn.functional as F
import torch

def compute_message_passing(edge_index, x, hop=2):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mp = MP()
    for _ in range(hop):
        x = mp.partition_propagate(edge_index, x=x, norm=norm, chunk_size=500, cuda=True)
    return x

def find_closest_class(x, meta_class_emb):
    # Normalize for accurate similarity calculation
    feature_tensor = F.normalize(x, dim=1)
    label_embeddings = F.normalize(meta_class_emb, dim=1)
    # feature_tensor = x
    # label_embeddings = meta_class_emb

    # Calculate cosine similarities (matrix of shape (N, C))
    similarities = torch.mm(feature_tensor, label_embeddings.T)

    # Find the index (class) with the maximum similarity for each feature vector 
    closest_classes = similarities.argmax(dim=1) 

    return closest_classes


def main(args):
    ## here, use only the dataset field
    dataset = load_one_tag_dataset(args.dataset, args.tag_data_path)
    x = dataset.x
    edge_index = dataset.edge_index
    x = compute_message_passing(edge_index, x, hop=3).cpu()
    meta_class_emb = dataset.meta_class_emb
    labels = dataset.y

    test_mask = dataset.test_mask[0]
    pred = find_closest_class(x[test_mask], meta_class_emb)
    acc = pred.eq(labels[test_mask]).sum().item() / test_mask.sum().item()
    print(f"Dataset: {args.dataset} Accuracy: {acc}")



if __name__ == '__main__':
    args = build_args()
    main(args)
