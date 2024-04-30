import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_optimizer, accuracy
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score



class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

@torch.no_grad()
def link_res(embedding, predictor, data, evaluator, batch_size):
    predictor.eval()
    h = embedding
    pos_train_edge = data.train_edge_index
    pos_valid_edge = data.pos_val_edge_index
    neg_valid_edge = data.neg_val_edge_index
    pos_test_edge = data.pos_test_edge_index
    neg_test_edge = data.neg_test_edge_index

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size):
        edge = pos_train_edge[:, perm].t()
        pos_train_preds += [predictor(h[edge[:, 0]], h[edge[:, 1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
        edge = pos_valid_edge[:, perm].t()
        pos_valid_preds += [predictor(h[edge[:, 0]], h[edge[:, 1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
        edge = neg_valid_edge[:, perm].t()
        neg_valid_preds += [predictor(h[edge[:, 0]], h[edge[:, 1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
        edge = pos_test_edge[:, perm].t()
        pos_test_preds += [predictor(h[edge[:, 0]], h[edge[:, 1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
        edge = neg_test_edge[:, perm].t()
        neg_test_preds += [predictor(h[edge[:, 0]], h[edge[:, 1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def link_linear_test(embedding, data, max_epoch, device, evaluator, mute = False):
    ## decode
    predictor = LinkPredictor(embedding.shape[1], 256, 1, 2, 0.0).to(device)
    optimizer = create_optimizer("adam", predictor, 0.01, 0.0)
    pos_train_edge = data.train_edge_index

    total_loss = total_examples = 0
    best_hits = 0
    best_val_hits = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)
    for epoch in epoch_iter:
        for perm in DataLoader(range(pos_train_edge.size(1)), batch_size=128, shuffle=True):
            optimizer.zero_grad()
            edge = pos_train_edge[:, perm].t().to(device)
            pos_out = predictor(embedding[edge[:, 0]], embedding[edge[:, 1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            neg_edge = torch.randint(0, embedding.size(0), edge.size(), dtype=torch.long, device=edge.device)
            neg_out = predictor(embedding[neg_edge[:, 0]], embedding[neg_edge[:, 1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += pos_loss.item() * pos_out.size(0)
            total_examples += pos_out.size(0)
        result = link_res(embedding, predictor, data, evaluator, 128)
        hits100 = result['Hits@100']
        test_hits = hits100[-1]
        val_hits = hits100[1]

        if val_hits > best_val_hits:
            best_val_hits = val_hits
            best_hits = test_hits
            # best_model = copy.deepcopy(predictor)
    print(f"--- TestHits@100: {best_hits:.4f}, Best ValHits@100: {best_val_hits:.4f} --- ")
    return best_hits, best_val_hits
    
def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list)/len(ap_list)

def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def eval_func(pred, labels, name = 'accuracy'):
    AVAILABLE = ['accuracy', 'f1', 'apr', 'auc']
    if name not in AVAILABLE:
        raise ValueError(f"eval_func should be one of {AVAILABLE}")
    if name == 'accuracy':
        return accuracy(pred, labels)
    elif name == 'f1':
        return f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
    elif name == 'apr':
        return eval_ap(labels.cpu().numpy(), pred.cpu().numpy())
    elif name == 'auc':
        return eval_rocauc(labels.cpu().numpy(), pred.cpu().numpy())

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

<<<<<<< HEAD
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
=======
def linear_test(embedding, data, max_epoch, device, m_name='accuracy', mute = False, eval_device = 'cpu'):
    lr = LogisticRegression(embedding.shape[1], data.num_classes).to(eval_device)
    optimizer = create_optimizer("adam", lr, 0.01, 0.0)
    
>>>>>>> c2a8a15edff2131d874b8ec5c89dc0d236c2c3bb

def linear_mini_batch_test(embedding, data, max_epoch, device, m_name='accuracy', mute = False, eval_device = 'cpu'):
    lr = LogisticRegression(embedding.shape[1], data.num_classes).to(eval_device)
    optimizer = create_optimizer("adam", lr, 0.005, 0.0)
    embedding = embedding.to(eval_device)

    train_mask = data.train_mask.to(eval_device)
    val_mask = data.val_mask.to(eval_device)
    test_mask = data.test_mask.to(eval_device)

    

    labels = []
    if hasattr(data, 'y'):
        labels = data.y.to(eval_device)
    else:
        for i in range(len(data.dataset)):
            labels.append(data.dataset[i].y)
        labels = torch.cat(labels, dim=0).to(eval_device)
    if labels.dim() < 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
<<<<<<< HEAD

    train_loader = DataLoader(EmbeddingDataset(embedding[train_mask], labels[train_mask]), batch_size=128, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(embedding[val_mask], labels[val_mask]), batch_size=128, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(embedding[test_mask], labels[test_mask]), batch_size=128, shuffle=False)

=======
>>>>>>> c2a8a15edff2131d874b8ec5c89dc0d236c2c3bb

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        for x, y in train_loader:
            lr.train()
            x = lr(x)
            is_labeled = y == y
            loss = criterion(x[is_labeled], y[is_labeled])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            lr.eval()
            pred = []
            true = []
            for x, y in val_loader:
                x = lr(x)
                pred.append(x)
                true.append(y)
            pred = torch.cat(pred, dim=0)
            true = torch.cat(true, dim=0)
            val_acc = eval_func(pred, true, m_name)
            val_loss = criterion(pred, true)
            pred = []
            true = []
            for x, y in test_loader:
                x = lr(x)
                pred.append(x)
                true.append(y)
            pred = torch.cat(pred, dim=0)
            true = torch.cat(true, dim=0)
            test_acc = eval_func(pred, true, m_name)
            test_loss = criterion(pred, true)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(lr)
    best_model.eval()
    with torch.no_grad():
        pred = best_model(embedding)
        estp_test_acc = eval_func(pred[test_mask], labels[test_mask], m_name)
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc, best_val_acc    


def linear_test(embedding, data, max_epoch, device, m_name='accuracy', mute = False, eval_device = 'cpu'):
    lr = LogisticRegression(embedding.shape[1], data.num_classes).to(eval_device)
    optimizer = create_optimizer("adam", lr, 0.01, 0.0)
    

    # data = data.to(device)
    embedding = embedding.to(eval_device)

    train_mask = data.train_mask.to(eval_device)
    val_mask = data.val_mask.to(eval_device)
    test_mask = data.test_mask.to(eval_device)
    labels = []
    if hasattr(data, 'y'):
        labels = data.y.to(eval_device)
    else:
        for i in range(len(data.dataset)):
            labels.append(data.dataset[i].y)
        labels = torch.cat(labels, dim=0).to(eval_device)
    if labels.dim() < 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        lr.train()
        x = lr(embedding)
        train_x = x[train_mask]
        train_labels = labels[train_mask]
        is_labeled = train_labels == train_labels
        loss = criterion(train_x[is_labeled], train_labels[is_labeled])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            lr.eval()
            pred = lr(embedding)
            val_pred = pred[val_mask]
            val_labels = labels[val_mask]
            is_labeled = val_labels == val_labels
            val_pred = val_pred[is_labeled]
            val_labels = val_labels[is_labeled]
            test_pred = pred[test_mask]
            test_labels = labels[test_mask]
            is_labeled = test_labels == test_labels
            test_pred = test_pred[is_labeled]
            test_labels = test_labels[is_labeled]
            val_acc = eval_func(pred[val_mask], labels[val_mask], m_name)
            val_loss = criterion(val_pred, val_labels)
            test_acc = eval_func(pred[test_mask], labels[test_mask], m_name)
            test_loss = criterion(test_pred, test_labels)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(lr)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(embedding)
        estp_test_acc = eval_func(pred[test_mask], labels[test_mask], m_name)
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc, best_val_acc




class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits