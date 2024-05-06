for d in "cora_link" "citeseer_link" "pubmed_link"
do
CUDA_VISIBLE_DEVICES=7 python run_cdm.py task_names ${d} d_multiple 1 d_min_ratio 1 lr 0.001 num_layers 5 num_epochs 30 dropout 0.15 exp_name ${d}_wandb
done

CUDA_VISIBLE_DEVICES=6 python run_cdm.py task_names "cora_link, citeseer_link, pubmed_link" lr 0.001 num_layers 5 num_epochs 30 dropout 0.15 exp_name "all_link_wandb"