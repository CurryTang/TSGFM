## GCC
## DGI
## GraphMAE
## Multi-cotrain

import numpy as np
import torch
from sklearn.metrics import f1_score

import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    Records
)
from graphmae.data_util import load_pretrain_dataset, load_train_segments, load_downstream_dataset
from graphmae.models import build_model
from graphmae.evaluation import node_classification_evaluation


def multi_eval(model, downstreams, device, lr_f, weight_decay_f, max_epoch_f, linear_prob, args, return_val = True):
    logging.info("start evaluating..")
    test_accs = []
    estp_accs = []
    val_accs = []
    estp_val_accs = []
    for i, g in enumerate(downstreams):
        graph = g.to(device)
        x = g.ndata["x"].to(device)
        label = g.ndata["y"].to(device)
        num_classes = label.max().item() + 1
        final_acc, estp_acc, val_acc, estp_val_acc = node_classification_evaluation(
            model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True, return_val = return_val)
        test_accs.append(final_acc)
        estp_accs.append(estp_acc)
        val_accs.append(val_acc)
        estp_val_accs.append(estp_val_acc)
    return test_accs, estp_accs, val_accs, estp_val_accs



def pretrain(model, dataloaders, d_map, downstreams, optimizer, max_epoch, device, scheduler, lr_f, weight_decay_f, max_epoch_f, linear_prob, args, logger=None):
    logging.info("start training..")
    
    epoch_iter = tqdm(range(max_epoch))
    records = Records(args.pre_train_datasets, args, args.initial_weight, args.min_ratio)

    for epoch in epoch_iter:
        model.train()
        loss_list = [[] for _ in range(len(args.pre_train_datasets))]
        loss_np = []

        for i, subgraph in enumerate(dataloaders):
            subgraph = subgraph.remove_self_loop().add_self_loop()
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["x"])
            data_idx = d_map[i]
            loss = loss * records.weight[data_idx]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list[d_map[i]].append(loss.item())

        if scheduler is not None:
            scheduler.step()
        
        for i, d in enumerate(args.pre_train_datasets):
            this_loss = loss_list[i]
            loss_np.append(np.mean(this_loss))
            print(f"Dataset {args.pre_train_datasets[i]} | loss: {np.mean(this_loss):.4f}")
        # epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
        
        test_accs, estp_accs, val_accs, estp_val_accs = multi_eval(
            model, downstreams, device, lr_f, weight_decay_f, max_epoch_f, linear_prob, args)
        for i, acc in enumerate(estp_val_accs):
            print(f"Epoch: {epoch} | Dataset {args.pre_train_datasets[i]} | val_acc: {acc:.4f} | test_acc: {estp_accs[i]:.4f}")
            records.update_metric(i, acc)
        records.update_weight()
        mean_val_acc = np.mean(estp_val_accs)
        mean_test_acc = np.mean(estp_accs)

        records.update_test_acc(mean_val_acc, mean_test_acc, estp_accs, estp_val_accs)
    
    best_test_acc = records.best_test_acc
    return model, best_test_acc


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

    rep = torch.ones(len(args.pre_train_datasets))
    graphs = load_pretrain_dataset(args.pre_train_datasets, rep, args)
    train_segments, segment_data_map, feature_shape = load_train_segments(graphs, args)
    args.num_features = feature_shape
    test_g = load_downstream_dataset(args.pre_train_datasets, args)
    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        if not load_model:
           model, best_test_acc = pretrain(model, train_segments, segment_data_map, test_g, optimizer, max_epoch, device, scheduler, lr_f, weight_decay_f, max_epoch_f, linear_prob, args, logger)
        model = model.cpu()

        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        acc_list.append(best_test_acc)


    for i in range(len(acc_list)):
        total = [acc[i] for acc in acc_list]
        print(f"Dataset {args.pre_train_datasets[i]} | Test Acc: {np.mean(total):.4f} | std: {np.std(total):.4f}")


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)