# Import the W&B Python Library and log into W&B
import wandb
# from supervised import main as sup_main
# from transductive_ssl import main as ssl_main
from sslmain import main as subgcon_main
from graphmae.utils import build_args, update_namespace

# 1: Define objective/training function
def objective(args):
    return subgcon_main(args)

def main(args):
    wandb.init(project="GFM")
    update_namespace(args, wandb.config)
    wandb.config.mode = args.mode
    best_avg_val, val_accs, test_accs  = objective(args)
    avg_test = sum(test_accs) / len(test_accs)
    wandb.log({"best_avg_val": best_avg_val, "avg_test": avg_test, "val_accs": val_accs, "test_accs": test_accs})

# 2: Define the search space
if __name__ == "__main__":
    args = build_args()
    if not args.not_same_pretrain_downstream:
        args.downstream_datasets = args.pre_train_datasets
    if args.method == 'graphmae':
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "avg_test"},
            "parameters": {
                "num_hidden": {"values": [128, 256, 512, 1024]},
                "num_heads": {"values": [1, 4, 8]},
                "num_out_heads": {"values": [1]},
                "activation": {"values": ["relu", "prelu", "elu"]},
                "in_drop": {"values": [0.0, 0.2, 0.5]},
                "attn_drop": {"values": [0.0, 0.2, 0.5]},
                "lr": {"values": [0.001, 0.005, 0.01]},
                "weight_decay": {"values": [0.0, 0.001, 5e-4, 1e-5]},
                "residual": {"values": [True, False]},
                "num_layers": {"values": [2, 3]},
                "scheduler": {"values": [True, False]},
                "norm": {"values": ["layernorm", "batchnorm", None]},
                'replace_rate': {"values": [0.0, 0.1, 0.2]},
                'mask_rate': {"values": [0.5, 0.75]},
                'drop_edge_rate': {"values": [0.0, 0.5]},
            },
        }
    elif args.method == 'dgi':
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "avg_test"},
            "parameters": {
                "num_hidden": {"values": [128, 256, 512, 1024]},
                "activation": {"values": ["relu", "prelu", "elu"]},
                "in_drop": {"values": [0.0, 0.2, 0.5]},
                "attn_drop": {"values": [0.0, 0.2, 0.5]},
                "lr": {"values": [0.001, 0.005, 0.01]},
                "weight_decay": {"values": [0.0, 0.001, 5e-4, 1e-5]},
                "num_layers": {"values": [2, 3]},
                "scheduler": {"values": [True, False]},
                "norm": {"values": ["layernorm", "batchnorm", None]},
                "lrtype": {"values": ["lambda", "cosine"]},
                "residual": {"values": [True, False]}
            },
        }
    elif args.mode == 'subgcon':
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "avg_test"},
            "parameters": {
                "num_hidden": {"values": [32, 64, 128, 256]},
                "in_drop": {"values": [0.0, 0.2, 0.5]},
                "attn_drop": {"values": [0.0, 0.1, 0.2, 0.5]},
                "lr": {"values": [0.001, 0.005, 0.01]},
                "weight_decay": {"values": [0.0, 1e-4, 2e-4, 0.05]},
                "weight_decay_f": {"values": [0.0, 1e-4, 2e-4]},
                "residual": {"values": [True, False]},
                "num_layers": {"values": [2, 3, 4]},
                "scheduler": {"values": [True, False]},
                "norm": {"values": ["layernorm", "batchnorm", None]},
                'replace_rate': {"values": [0.0, 0.1, 0.2]},
                'mask_rate': {"values": [0.5, 0.75]},
                'drop_edge_rate': {"values": [0.0, 0.5]},
                'alpha_l': {"values": [2, 3, 4, 5]},
                'lr_f': {"values": [0.001, 0.005, 0.025]},
                'remask_method': {"values": ['fixed', 'random']},
                'lam': {"values": [0.1, 1,  5, 10]},
                'momentum': {'values': [0., 0.96, 1.]},
            }
        }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"GFM-{args.pre_train_datasets}-{args.encoder}"[:75])

    wandb.agent(sweep_id, function=lambda: main(args), count=args.count)