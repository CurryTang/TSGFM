# Import the W&B Python Library and log into W&B
import wandb
from supervised import main as sup_main
from transductive_ssl import main as ssl_main
from graphmae.utils import build_args, update_namespace

# 1: Define objective/training function
def objective(args):
    if args.mode == 'sup':
        return sup_main(args)
    elif args.mode == 'ssl':
        return ssl_main(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def main(args):
    wandb.init(project="GFM")
    update_namespace(args, wandb.config)
    wandb.config.mode = args.mode
    score, std = objective(args)
    wandb.log({"acc": score, 'std': std})

# 2: Define the search space
if __name__ == "__main__":
    args = build_args()
    if args.mode == 'sup':
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "score"},
            "parameters": {
                "num_hidden": {"values": [16, 32, 64, 128, 256]},
                "num_heads": {"values": [1, 4, 8]},
                "num_out_heads": {"values": [1]},
                "activation": {"values": ["relu", "prelu", "elu"]},
                "in_drop": {"values": [0.0, 0.2, 0.5]},
                "attn_drop": {"values": [0.0, 0.2, 0.5]},
                "lr": {"values": [0.001, 0.005, 0.01]},
                "weight_decay": {"values": [0.0, 0.001, 0.005]},
                "residual": {"values": [True, False]},
                "num_layers": {"values": [2, 3]},
                "scheduler": {"values": [True, False]},
                "norm": {"values": ["layernorm", "batchnorm", None]},
            },
        }
    elif args.mode == 'ssl':
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "maximize", "name": "score"},
            "parameters": {
                "num_hidden": {"values": [128, 256, 512, 1024]},
                "num_heads": {"values": [1, 4, 8]},
                "num_out_heads": {"values": [1]},
                "activation": {"values": ["relu", "prelu", "elu"]},
                "in_drop": {"values": [0.0, 0.2, 0.5]},
                "attn_drop": {"values": [0.0, 0.2, 0.5]},
                "lr": {"values": [0.001, 0.005, 0.01]},
                "weight_decay": {"values": [0.0, 0.001, 0.005]},
                "residual": {"values": [True, False]},
                "num_layers": {"values": [2, 3]},
                "scheduler": {"values": [True, False]},
                "norm": {"values": ["layernorm", "batchnorm", None]},
                'replace_rate': {"values": [0.0, 0.1, 0.2]},
                'mask_rate': {"values": [0.5, 0.75]},
                'drop_edge_rate': {"values": [0.0, 0.5]},
                'alpha_l': {"values": [1, 2, 3]}
            }
        }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"GFM-{args.dataset}-{args.pre_train_datasets}-{args.encoder}")

    wandb.agent(sweep_id, function=lambda: main(args), count=args.count)