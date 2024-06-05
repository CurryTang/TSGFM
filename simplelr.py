from graphmae.data_util import load_one_tag_dataset, unify_dataset_loader
from graphmae.utils import build_args
from torch_geometric.utils import to_undirected, remove_self_loops, degree
from graphllm.utils import MP
import torch.nn.functional as F
import torch
from ogb.linkproppred.evaluate import Evaluator
from task_constructor import LabelPerClassSplit

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

def update_few_shot_train_mask(data):
    num_class = data.y.max().item() + 1
    total_num = data.y.size(0)
    splitter = LabelPerClassSplit(num_labels_per_class=3, num_valid=3, num_test=3)
    train_mask, _, _ = splitter(data, total_num)
    data.train_mask = train_mask


def compute_class_emb(few_shot_train_mask, data):
    class_embs = []
    for i in range(data.y.max().item() + 1):
        class_embs.append(data.x[few_shot_train_mask & (data.y == i)].mean(dim=0))
    return torch.stack(class_embs, dim=0)


def find_closest_class(x, meta_class_emb):
    # Normalize for accurate similarity calculation
    #feature_tensor = F.normalize(x, dim=1)
    # label_embeddings = F.normalize(meta_class_emb, dim=1)
    feature_tensor = x
    label_embeddings = meta_class_emb
    # feature_tensor = x
    # label_embeddings = meta_class_emb

    # Calculate cosine similarities (matrix of shape (N, C))
    similarities = torch.mm(feature_tensor, label_embeddings.T)

    # Find the index (class) with the maximum similarity for each feature vector 
    closest_classes = similarities.argmax(dim=1) 

    return closest_classes


def pair_cosine_similarity(emb, edge_index):
    source_idx = edge_index[0]
    target_idx = edge_index[1]
    source_emb = emb[source_idx]
    target_emb = emb[target_idx]
    cosine_sim = F.cosine_similarity(source_emb, target_emb, dim=1)
    return cosine_sim

def main(args):
    ## here, use only the dataset field
    if 'link' not in args.dataset:
        dataset = load_one_tag_dataset(args.dataset, args.tag_data_path)
        x = dataset.x
        edge_index = dataset.edge_index
        x = compute_message_passing(edge_index, x, hop=3).cpu()
        if not args.fewshot:
            meta_class_emb = dataset.meta_class_emb
        else:
            update_few_shot_train_mask(dataset)
            meta_class_emb = compute_class_emb(dataset.train_mask, dataset)
            meta_class_emb = meta_class_emb + dataset.meta_class_emb
        labels = dataset.y

        test_mask = dataset.test_mask
        pred = find_closest_class(x[test_mask], meta_class_emb)
        acc = pred.eq(labels[test_mask]).sum().item() / test_mask.sum().item()
        print(f"Dataset: {args.dataset} Accuracy: {acc}")
    else:
        evaluator = Evaluator(name='ogbl-ppa')
        dataset = unify_dataset_loader([args.dataset], args)[0]
        x = dataset.x
        edge_index = dataset.edge_index
        x = compute_message_passing(edge_index, x, hop=3).cpu()
        pos_logits = pair_cosine_similarity(x, dataset.pos_test_edge_index)
        neg_logits = pair_cosine_similarity(x, dataset.neg_test_edge_index)
        res = evaluator.eval({
            'y_pred_pos': pos_logits,
            'y_pred_neg': neg_logits
        })['hits@100']
        print(f"Dataset: {args.dataset} Hits@100: {res}")



if __name__ == '__main__':
    args = build_args()
    main(args)
