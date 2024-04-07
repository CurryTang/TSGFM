#!/bin/bash

max_len=4096
sample_size=10
model="mistral"
task="nc"
dataset="cora-citeseer-pubmed-arxiv-arxiv23"
bs="16"
emb="sbert"
cache_dir="./llmcheckpoint"


use_hop=4
template="HO"
projector_type="linear"
prefix=llaga-mistral-7b-hf-${emb}-${use_hop}-hop-token-${projector_type}-projector
model_base=mistralai/Mistral-7B-v0.1
mode="mistral_instruct"

## determine parameters based on your gpu types
gpu_info=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)

# Check for presence of V100 or A100
if [[ $gpu_info == *"Tesla V100"* ]]; then
    echo "Tesla V100 detected (GPU 0)"
    fp16=True
    bf16=False
    tf32=False
elif [[ $gpu_info == *"Tesla A100"* ]]; then
    echo "Tesla A100 detected (GPU 0)"
    fp16=False
    bf16=True
    tf32=True
else
    echo "Neither Tesla V100 nor A100 detected (GPU 0)"
    exit
fi


echo "PREFIX:  ${prefix}"

WANDB_DISABLED=true
singularity exec --nv /mnt/home/chenzh85/pytorch.sif python3 graphllm.py \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  ${cache_dir} \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--fp16 ${fp16} \
--bf16 ${bf16} \
--output_dir  ./checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 ${tf32} \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template}