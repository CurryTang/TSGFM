import logging
import os
import numpy as np
from tqdm import tqdm

import torch

from graphmae.utils import (
    WandbLogger,
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
    show_occupied_memory,
)
from graphmae.models import build_model
from graphmae.lc_sampler import setup_training_dataloder, setup_multiple_training_dataloder, setup_saint_dataloader, drop_edge
from graphmae.evaluation import linear_probing_minibatch, finetune, linear_probing_full_batch
from graphmae.data_util import load_downstream_dataset, load_pretrain_dataset, mask_to_split_idx
from bgrl.bgrl import build_bgrl_model, drop_feature_edges, CosineDecayScheduler, compute_representations
from torch import cosine_similarity
import copy
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def evaluate(
    args, model, 
    graph, feats, labels, 
    split_idx, 
    lr_f, weight_decay_f, max_epoch_f, 
    linear_prob=True, 
    device=0, 
    batch_size=256, 
    logger=None,
    shuffle=True
):
    logging.info("Using `lc` for evaluation...")
    num_train, num_val, num_test = [split_idx[k].shape[0] for k in ["train", "valid", "test"]]
    print(num_train,num_val,num_test)

    num_classes = labels.max().item() + 1

    train_g_idx = split_idx["train"]
    val_g_idx = split_idx["valid"]
    test_g_idx = split_idx["test"]
    print(f"num_train: {num_train}, num_val: {num_val}, num_test: {num_test}")

    
    if args.method == 'graphmae':
        if linear_prob:
            final_acc, estp_acc = linear_probing_full_batch(model, graph, feats, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=True)
        else:
            raise NotImplementedError("Not implemented yet")
    else:
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        final_acc, estp_acc = linear_probing_full_batch(tmp_encoder, graph, feats, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=True, mode='bgrl')
    return estp_acc


def pretrain(args, model, graphs, downstream_datas, max_epoch, device, use_scheduler, lr, weight_decay, lr_f, weight_decay_f, max_epoch_f, batch_size_f, optimizer="adam", drop_edge_rate=0, eval_period = 1):
    logging.info("start training..")

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)    

    dataloaders = setup_saint_dataloader(
        args, graphs
    )

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
        if args.method == 'bgrl':
            scheduler = CosineDecayScheduler(args.lr, args.warmup_steps, args.max_epoch)
            mm_scheduler = CosineDecayScheduler(1-args.momentum, 0, args.max_epoch)
        else:
            mm_scheduler = None

    for epoch in range(max_epoch):
        for dataloader in dataloaders:
            epoch_iter = tqdm(dataloader)
            losses = []
            # assert (graph.in_degrees() > 0).all(), "after loading"

            for batch_g in epoch_iter:
                model.train()
                if args.method == 'graphmae':
                    if drop_edge_rate > 0:
                        x = batch_g.ndata['x']
                        targets = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
                        batch_g = batch_g.to(device)
                        drop_g1, drop_g2 = drop_edge(batch_g, drop_edge_rate)
                        drop_g1 = drop_g1.to(device)
                        drop_g2 = drop_g2.to(device)
                        loss = model(batch_g, x, targets, epoch, drop_g1, drop_g2)
                    else:
                        x = batch_g.ndata['x']
                        targets = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
                        batch_g = batch_g.to(device)
                        loss = model(batch_g, x, targets, epoch)
                else:
                    drop_g1 = drop_feature_edges(batch_g, args.drop_feat_rate, args.drop_edge_rate)
                    drop_g2 = drop_feature_edges(batch_g, args.drop_feat_rate_2, args.drop_edge_rate_2)

                    lr = scheduler.get(epoch)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    
                    mm = 1 - mm_scheduler.get(epoch)
                    optimizer.zero_grad()

                    q1, y2 = model(drop_g1, drop_g2)
                    q2, y1 = model(drop_g2, drop_g1)

                    loss = 2 - cosine_similarity(q1, y2.detach(), dim = -1).mean() - cosine_similarity(q2, y1.detach(), dim = -1).mean()
                    loss.backward()
                    optimizer.step()
                    model.update_target_network(mm)
                if args.method == 'graphmae':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    optimizer.step()

                epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
                losses.append(loss.item())
            
            if scheduler is not None and args.method == 'graphmae':
                scheduler.step()

            ## torch.save(model.state_dict(), os.path.join(model_dir, model_name))

            print(f"# Epoch {epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB")

        if epoch % eval_period == 0:
            downstream_task(args, model, device, downstream_datas, None, lr_f, weight_decay_f, max_epoch_f, batch_size_f, 0, final_eval = False)

    return model


def downstream_task(args, model, device, downstream_datas, final_accs,
                    lr_f, weight_decay_f, max_epoch_f,batch_size_f, seed, final_eval = False):
    if args.method == 'graphmae':
        eval_model = build_model(args)
        eval_model.load_state_dict(model.state_dict())
        eval_model.to(device)
    else:
        eval_model = model

    for graph, graph_name in zip(downstream_datas, args.downstream_datasets):
        if final_eval:
            if final_accs.get(graph_name) is None:
                final_accs[graph_name] = []
        feats = graph.ndata["x"]
        labels = graph.ndata["y"]
        split_idx = mask_to_split_idx(graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"])
        print(f"features size : {feats.shape[1]}")
        logging.info("start evaluation...")
        final_acc = evaluate(
            args, eval_model, graph, feats, labels, split_idx,
            lr_f, weight_decay_f, max_epoch_f, 
            device=device, 
            batch_size=batch_size_f, 
            linear_prob=True,
            shuffle=False if graph_name == "ogbn-papers100M" else True
        )
        
        if final_eval:
            final_accs[graph_name].append(float(final_acc))

        print(f" Seed: {seed} | Dataset: {graph_name} | TestAcc: {final_acc:.4f}")

    del eval_model
    return final_accs


def main(args):
    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    encoder = args.encoder
    decoder = args.decoder
    num_hidden = args.num_hidden
    drop_edge_rate = args.drop_edge_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    no_pretrain = args.no_pretrain
    logs = args.logging
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    batch_size_f = args.batch_size_f
    sampling_method = args.sampling_method
    ego_graph_file_path = args.ego_graph_file_path
    data_dir = args.data_dir

    n_procs = torch.cuda.device_count()
    optimizer_type = args.optimizer
    label_rate = args.label_rate
    lam = args.lam
    full_graph_forward = hasattr(args, "full_graph_forward") and args.full_graph_forward and not linear_prob

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    set_random_seed(0)
    print(args)
    
    datasets = load_pretrain_dataset(
        args.pre_train_datasets, args.initial_weight, args 
    )

    downstream_datas = load_downstream_dataset(
        args.downstream_datasets, args
    )

    if logs:
        logger = WandbLogger(log_path=f"{args.pre_train_datasets}_loss_{loss_fn}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}", project="GraphMAE2", args=args)
    else:
        logger = None
    model_name = f"{encoder}_{decoder}_{num_hidden}_{num_layers}_{dataset_name}_{args.mask_rate}_{num_hidden}_checkpoint.pt"

    args.num_features = datasets[0]['g'].ndata["x"].shape[1]
    
    if args.method == 'graphmae':
        model = build_model(args)
    else:
        model = build_bgrl_model(args)

    if not args.no_pretrain:
        # ------------- pretraining starts ----------------
        if not load_model:
            logging.info("---- start pretraining ----")
            model = pretrain(args, model, datasets, downstream_datas, max_epoch=max_epoch, device=device, use_scheduler=use_scheduler, lr=lr, 
            weight_decay=weight_decay, lr_f = lr_f, weight_decay_f=weight_decay_f, max_epoch_f=max_epoch_f, batch_size_f=batch_size_f, drop_edge_rate=drop_edge_rate,   optimizer=optimizer_type)
        
            model = model.cpu()
            logging.info(f"saving model to {model_dir}/{model_name}...")
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        # ------------- pretraining ends ----------------   

        if load_model:
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_path)))
            logging.info(f"Loading Model from {args.checkpoint_path}...")
    else:
        logging.info("--- no pretrain ---")

    model = model.to(device)
    model.eval()

    logging.info("---- start finetuning / evaluation ----")

    final_accs = {}
    for i,_ in enumerate(seeds):
        print(f"####### Run seed {seeds[i]}")
        set_random_seed(seeds[i])
        
        downstream_task(args, model, device, downstream_datas, final_accs, lr_f, weight_decay_f, max_epoch_f, batch_size_f, seeds[i], final_eval = True)
            
    print("Final acc")
    data_mean = []
    for key, val in final_accs.items():
        mean_acc = np.mean(val)
        std_acc = np.std(val)
        print(f"Dataset {key}: {mean_acc:.4f} +- {std_acc:.4f}")
        data_mean.append(mean_acc)
    
    return_val = np.mean(data_mean)
    print(f"Mean: {np.mean(data_mean):.4f}")
    
    if logger is not None:
        logger.finish()
    
    return return_val


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)
    main(args)
    