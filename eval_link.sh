CUDA_VISIBLE_DEVICES=0 python run_cdm.py task_names cora_link d_multiple 5 d_min_ratio 5 lr 0.0001 num_layers 7 num_epochs 20  exp_name cora_wandb


CUDA_VISIBLE_DEVICES=0 python run_cdm.py task_names citeseer_link d_multiple 5 d_min_ratio 5 lr 0.0001 num_layers 7 num_epochs 20  exp_name citeseer_wandb


CUDA_VISIBLE_DEVICES=0 python run_cdm.py task_names pubmed_link d_multiple 1 d_min_ratio 1 lr 0.0001 num_layers 7 num_epochs 20  exp_name pubmed_wandb


CUDA_VISIBLE_DEVICES=0 python run_cdm.py task_names "cora_link, citeseer_link, pubmed_link" lr 0.0001 num_layers 7 num_epochs 20  d_multiple 3,3,1 d_min_ratio 1,1,0.5 exp_name "all_link_wandb"