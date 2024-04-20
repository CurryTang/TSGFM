#!/bin/bash

dataset=${1:-"cora"} #test dataset
task=${2:-"lp"} #test task
model_path=${3:-"/root/paddlejob/workspace/env_run/MyOFA/checkpoints/cora/llaga-mistral-7b-hf-sbert-4-hop-token-linear-projector_nc"}
pretrain_data=${4:-"cora.3-citeseer.3"}
model_base="../Mistral-7B-v0.1" #meta-llama/Llama-2-7b-hf
mode="mistral_instruct" # use 'llaga_llama_2' for llama and "v1" for others
emb="sbert"
use_hop=4
sample_size=10
template="HO" # or ND
output_path="./checkpoints/${dataset}_${task}_${pretrain_data}"

python3.8 eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir "./llmcheckpoint" \
--template ${template} \
--start "-1" \
--end "2000"

