#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    cfg_overrides=$1
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python run_cdm.py ${cfg_overrides}"
    common_params="${cfg_overrides}"

    echo "Run program: ${main}"


    script="sbatch ${slurm_directive} --output \"${cfg_overrides}\" run/wrapper.sb ${main}"
    echo $script
    eval $script
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

slurm_directive="--time=0-8:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=6"
run_repeats "--override ./eval_all_node_oneforall.yaml"


# for dataset in "amazonratings" "bookchild" "bookhis" "elecomp" "elephoto" "sportsfit" "wikics"
# for dataset in "citeseer_node"
# do 
#     run_repeats "task_names ${dataset} num_layers 5 num_epochs 100 d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK none batch_size 64 emb_dim 64"
#     run_repeats "task_names ${dataset} num_layers 7 num_epochs 100 d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK none batch_size 64 emb_dim 64"
# done

# slurm_directive="--time=0-1:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
# for dataset in "cora" "citeseer" "pubmed"
# do 
#     # run_repeats "/mnt/home/chenzh85/graphlang/PyGFM/configs/gnn/planetoid.yaml" "dataset.name ${dataset}"
#     echo "Running ${dataset} node classification"
#     run_repeats "task_names ${dataset}_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64 emb_dim 64"
#     run_repeats "task_names ${dataset}_node num_layers 7 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64 emb_dim 64"
#     run_repeats "task_names ${dataset}_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.001 JK none batch_size 64 emb_dim 64"
#     run_repeats "task_names ${dataset}_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64 emb_dim 128"
#     run_repeats "task_names ${dataset}_node num_layers 7 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64 emb_dim 128"
# done 

# slurm_directive="--time=0-3:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
# run_repeats "task_names arxiv23 num_layers 5 num_epochs 15 d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK last batch_size 256 emb_dim 64"
# run_repeats "task_names arxiv23 num_layers 5 num_epochs 20 d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK last batch_size 256 emb_dim 64"
# for dataset in "arxiv" "arxiv23"
# do 
#     for num_layers in 5 6 8
#     do
#         for num_epochs in 5 10 20
#         do 
#             for lr in 0.01 0.001 0.0001
#             do  
#                 for jk in "last" "none"
#                 do 
#                     for emb_dim in 64 128 256
#                     do 
#                         # run_repeats "/mnt/home/chenzh85/graphlang/PyGFM/configs/gnn/planetoid.yaml" "dataset.name ${dataset}"
#                         echo "Running ${dataset} node classification"
#                         echo "params num_layers ${num_layers} num_epochs ${num_epochs} lr ${lr} JK ${jk} emb_dim ${emb_dim}"
#                         run_repeats "task_names ${dataset} num_layers ${num_layers} num_epochs ${num_epochs} d_multiple 1.0 d_min_ratio 1.0 lr ${lr} JK ${jk} batch_size 256 emb_dim ${emb_dim}"
#                     done
#                 done
#             done
#         done 
#     done 
# done 








