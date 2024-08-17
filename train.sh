# CUDA_VISIBLE_DEVICES=2 python3 gpreprocess.py --dataset_name arxiv --gpu 0 --type pretrain --max_length 100
# CUDA_VISIBLE_DEVICES=2 python3 gpreprocess.py --dataset_name cora --gpu 0 --type pretrain  --max_length 1000
CUDA_VISIBLE_DEVICES=3 python3 gpreprocess.py --dataset_name bookhis --gpu 0 --type pretrain --max_length 300
# CUDA_VISIBLE_DEVICES=2 python3 gpreprocess.py --dataset_name amazonratings --gpu 0 --type pretrain
# CUDA_VISIBLE_DEVICES=2 python3 gpreprocess.py --dataset_name cora --gpu 0 --type prompt --max_length 1000
CUDA_VISIBLE_DEVICES=3 python3 gpreprocess.py --dataset_name bookhis --gpu 0 --type prompt --max_length 300
# CUDA_VISIBLE_DEVICES=2 python3 gpreprocess.py --dataset_name amazonratings --gpu 0 --type prompt

# CUDA_VISIBLE_DEVICES=2 python3 gpretrain.py --dataset_name cora --hiddensize_gnn 128 --hiddensize_fusion 128 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15
CUDA_VISIBLE_DEVICES=3 python3 gpretrain.py --dataset_name bookhis --hiddensize_gnn 128 --hiddensize_fusion 128 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15
# CUDA_VISIBLE_DEVICES=2 python3 gpretrain.py --dataset_name amazonratings --hiddensize_gnn 128 --hiddensize_fusion 128 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15


# CUDA_VISIBLE_DEVICES=2 python3 gfinetune.py --dataset_name cora --gpu 0 --metric acc --save_path save_models/cora/128_128_SAGE_2_32_0.0005_0.001_15_10/ 
CUDA_VISIBLE_DEVICES=3 python3 gfinetune.py --dataset_name bookhis --gpu 0 --metric acc --save_path save_models/bookhis/128_128_SAGE_2_32_0.0005_0.001_15_10/
# CUDA_VISIBLE_DEVICES=2 python3 gfinetune.py --dataset_name amazonratings --gpu 0 --metric acc --save_path save_models/amazonratings/128_128_SAGE_2_32_0.0005_0.001_15_10/

# CUDA_VISIBLE_DEVICES=2 python3 gpretrain.py --dataset_name arxiv --hiddensize_gnn 64 --hiddensize_fusion 64 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15