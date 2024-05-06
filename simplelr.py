from graphmae.data_util import load_one_tag_dataset
from graphmae.utils import build_args
from torch_geometric.utils import to_undirected, remove_self_loops, degree
from graphllm.utils import MP


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


def main(args):
    ## here, use only the dataset field
    dataset = load_one_tag_dataset(args.dataset, args.tag_data_path)
    x = dataset.x
    edge_index = dataset.edge_index
    x = compute_message_passing(edge_index, x, hop=2)
    meta_class_emb = dataset.meta_class_emb
    


if __name__ == '__main__':
    args = build_args()
    main(args)
