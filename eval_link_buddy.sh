# link=("cora" "citeseer" "pubmed" "arxiv" "arxiv23")

# #!/bin/bash

# function get_gpu_with_most_memory() {
#   gpu_data=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader) 

#   max_memory=-1
#   max_gpu_id=-1

#   while IFS=',' read -r gpu_id free_memory; do
#     # Extract the numerical part of the free memory
#     free_memory_num=$(echo $free_memory | cut -d' ' -f1)  

#     if [[ $gpu_id -gt 1 && $free_memory_num -gt $max_memory ]]; then
#       max_memory=$free_memory_num
#       max_gpu_id=$gpu_id
#     fi
#   done <<< "$gpu_data"

#   echo $max_gpu_id
# }

# echo "Single train"
# for d in "cora-link" "citeseer-link" "pubmed-link" "arxiv-link" "arxiv23-link"
# do 
#     if [ "${d}" == "arxiv-link" ]; then
#         layers=3
#     else
#         layers=2
#     fi
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=2 python3 fulllink.py --pre_train_datasets ${d} --encoder mlp --num_layers ${layers} --num_hidden 128  & 
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=3 python3 fulllink.py --pre_train_datasets ${d} --encoder sign --num_layers ${layers} --num_hidden 128 & 
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=4 python3 fulllink.py --pre_train_datasets ${d} --encoder gcn --num_layers ${layers} --num_hidden 128 &
#     gpu=$(get_gpu_with_most_memory)
#     base_name="${d%-*}" 
#     CUDA_VISIBLE_DEVICES=5 python3 linkpred.py --pre_train_datasets ${base_name} --model BUDDY --cache_subgraph_features --max_hash_hops ${layers} &
# done 



# for d in "bookhis-link" "bookchild-link" "sportsfit-link" "products-link" "elecomp-link" "elephoto-link"
# do 
#     layers=3
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=6 python3 fulllink.py --pre_train_datasets ${d} --encoder mlp --num_layers ${layers} --num_hidden 128  & 
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=7 python3 fulllink.py --pre_train_datasets ${d} --encoder sign --num_layers ${layers} --num_hidden 128 & 
#     gpu=$(get_gpu_with_most_memory)
#     CUDA_VISIBLE_DEVICES=8 python3 fulllink.py --pre_train_datasets ${d} --encoder gcn --num_layers ${layers} --num_hidden 128 &
#     gpu=$(get_gpu_with_most_memory)
#     base_name="${d%-*}"
#     CUDA_VISIBLE_DEVICES=9 python3 linkpred.py --pre_train_datasets ${base_name} 
#     --model BUDDY --cache_subgraph_features --max_hash_hops ${layers} &
# done 

# wait 


CUDA_VISIBLE_DEVICES=9 python3 fulllink.py --pre_train_datasets "cora-link" "citeseer-link" "arxiv-link" "arxiv23-link" --encoder gcn --num_layers 3 --num_hidden 128 --batch_size 512 &> gcn_co_citation.log 

CUDA_VISIBLE_DEVICES=9 python3 fulllink.py --pre_train_datasets "bookhis-link" "bookchild-link" "sportsfit-link" "products-link" "elecomp-link" "elephoto-link" --encoder gcn --num_layers 3 --num_hidden 128 --batch_size 512 &> gcn_co_commerce.log 

CUDA_VISIBLE_DEVICES=9 python3 fulllink.py --pre_train_datasets "cora-link" "citeseer-link" "pubmed-link" "arxiv-link" "arxiv23-link" "bookhis-link" "bookchild-link" "sportsfit-link" "products-link" "elecomp-link" "elephoto-link" --encoder gcn --num_layers 3 --num_hidden 128 --batch_size 512 &> gcn_co_all.log 