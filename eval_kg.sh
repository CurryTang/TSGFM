for d in "WN18RR" "FB15K237" "wikics"
do
CUDA_VISIBLE_DEVICES=7 python run_cdm.py task_names ${d} d_multiple 1 d_min_ratio 1 lr 0.001 num_layers 3 num_epochs 30 dropout 0.15
done