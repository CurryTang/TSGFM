import logging
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.data_util import load_downstream_dataset
from graphmae.utils import create_norm, Records
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from graphmae.models.gcn import GCN
from graphmae.models.gat import GAT
from graphmae.models.mlp import MLP



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)



def build_model(args, in_dim, out_dim):
    ## for supervised baseline, we only have encoder
    ## encoding should be False
    if args.encoder == 'gat':
        args.num_hidden = args.num_hidden // args.num_heads
        return GAT(
            in_dim, 
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=create_norm(args.norm),
            concat_out=True,
            encoding=False
        ) 
    elif args.encoder == 'gcn':
        return GCN(
            in_dim,
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            dropout=args.in_drop,
            activation=args.activation,
            residual=args.residual,
            norm=create_norm(args.norm),
            encoding=False
        )
    elif args.encoder == 'mlp':
        return MLP(
            in_dim,
            args.num_hidden,
            out_dim,
            num_layers=args.num_layers,
            dropout=args.in_drop,
            activation=args.activation,
            norm=create_norm(args.norm),
            encoding=False
        )
    
class MultiheadModel(torch.nn.Module):
    def __init__(self, args, in_dim, encoder_space_dim):
        super(MultiheadModel, self).__init__()
        self.encoder = build_model(args, in_dim, encoder_space_dim)
        self.heads = []
        self.Gs = []
        for d in args.pre_train_datasets:
            G = load_downstream_dataset([d], args)[0]
            dim = G.ndata['y'].max().item() + 1
            t_head = torch.nn.Linear(encoder_space_dim, dim)
            self.heads.append(t_head)
    
    def forward(self, G, x, i):
        x = self.encoder(G, x)
        x = self.heads[i](x)
        return x


@torch.no_grad()
def eval(model, graph, feat, device, labels, mask):
    logging.info("start evaluating..")
    graph = graph.to(device)
    x = feat.to(device)
    labels = labels.to(device)

    model.eval()
    output = model(graph, x)
    pred = output.argmax(dim=1)
    acc = (pred[mask] == labels[mask]).float().mean().item()
    return acc




        
    
def multi_train(model, dataset, graphs, optimizer, max_epoch, device, scheduler):
    logging.info("start training..")
    train_accs = []
    val_accs = []
    test_accs = []
    record = Records(dataset, args, initial_weight=args.initial_weight, min_ratio=args.min_ratio)
    for e in range(max_epoch):
        for i, g in enumerate(graphs):
            graph = g.to(device)
            x = g.ndata["x"].to(device)
            label = g.ndata["y"].to(device)
            train_mask = g.ndata["train_mask"]
            val_mask = g.ndata["val_mask"]
            test_mask = g.ndata["test_mask"]
            model.train()
            optimizer.zero_grad()
            output = model(graph, x, i)
            loss = record.weight[i] * torch.nn.CrossEntropyLoss()(output[train_mask], label[train_mask])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        train_accs, val_accs, test_accs = multi_eval(model, graphs, device)
            # val_acc = eval(model, graph, x, device, label, val_mask)
            # test_acc = eval(model, graph, x, device, label, test_mask)
        for i, d in enumerate(dataset):
            train_acc = train_accs[i]
            val_acc = val_accs[i]
            test_acc = test_accs[i]
            logging.info(f"Dataset {d}: train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
            record.update_metric(i, val_acc)
    return train_accs, val_accs, test_accs

def multi_eval(model, graphs, device):
    logging.info("start evaluating..")
    train_accs = []
    val_accs = []
    test_accs = []
    for i, g in enumerate(graphs):
        graph = g.to(device)
        x = g.ndata["x"].to(device)
        label = g.ndata["y"].to(device)
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        model.eval()
        output = model(graph, x, i)
        pred = output.argmax(dim=1)
        train_acc = (pred[train_mask] == label[train_mask]).float().mean().item()
        train_accs.append(train_acc)
        val_acc = (pred[val_mask] == label[val_mask]).float().mean().item()
        val_accs.append(val_acc)
        test_acc = (pred[test_mask] == label[test_mask]).float().mean().item()
        test_accs.append(test_acc)
    return train_accs, val_accs, test_accs


def train(model, graph, feat, optimizer, max_epoch, device, scheduler, label):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    label = label.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    epoch_iter = tqdm(range(max_epoch))

    best_val = 0
    best_test = 0
    best_output = None

    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        output = model(graph, x)
        loss = torch.nn.CrossEntropyLoss()(output[train_mask], label[train_mask])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        val_acc = eval(model, graph, x, device, label, val_mask)
        test_acc = eval(model, graph, x, device, label, test_mask)
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_output = output

    # return best_model
    best_pred = best_output.argmax(dim=1)
    return best_val, best_test, best_pred, best_output


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    # graph, (num_features, num_classes) = load_dataset(dataset_name)
    if args.multi_sup_mode:
        graphs = load_downstream_dataset(args.pre_train_datasets, args)
    else:
        graphs = load_downstream_dataset([args.dataset], args)
    G = graphs[0]
    args.num_features = G.ndata['x'].shape[1]
    args.num_classes = G.ndata['y'].max().item() + 1
    acc_list = []
    estp_acc_list = []
    preds = []
    output = []
    tests = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args, args.num_features, args.num_classes)
        if args.multi_sup_mode:
            model = MultiheadModel(args, args.num_features, args.num_hidden)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)
        labels = G.ndata["y"]


        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = G.ndata["x"]
        if not load_model:
            if args.multi_sup_mode:
                train_accs, val_accs, test_accs = multi_train(model, args.pre_train_datasets, graphs, optimizer, max_epoch, device, scheduler)
            else:
                best_val, best_test, best_pred, best_output = train(model, G, x, optimizer, max_epoch, device, scheduler, labels)
        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        if not args.multi_sup_mode:
            acc_list.append(best_test)
            preds.append(best_pred)
        else:
            tests.append(test_accs)
        
        
    if args.multi_sup_mode:
        faccs = []
        for i in range(len(args.pre_train_datasets)):
            name = args.pre_train_datasets[i]
            acc = [t[i] for t in tests]
            print(f"Dataset {name}: {np.mean(acc):.4f}±{np.std(acc):.4f}")
            faccs.append(np.mean(acc))
        facc = np.mean(faccs)
        facc_std = np.std(faccs)
    else:
        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        torch.save(preds, f"{dataset_name}_{args.encoder}_preds.pt")
    if args.multi_sup_mode:
        return facc, facc_std
    else:
        return final_acc, final_acc_std


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg != "":
        args = load_best_configs(args, args.use_cfg)
    print(args)
    main(args)