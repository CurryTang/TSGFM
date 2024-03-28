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

function run_srepeats {
    cfg_overrides=$1
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python supervised.py ${cfg_overrides}"
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



slurm_directive="--time=0-1:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
for dataset in "cora" "citeseer" "pubmed"
do 
    for num_layers in 1 2 4 6 8 12 16
    do 
        echo "Running ${dataset} node classification"
        # run_repeats "task_names ${dataset}_node num_layers ${num_layers} num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 256 emb_dim 128"
        run_srepeats "--dataset ${dataset} --num_layers ${num_layers} --encoder gcn --num_hidden 64 --lr 0.01 --in_drop 0.5 --weight_decay 5e-4"
    done
done 

slurm_directive="--time=0-1:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
for dataset in "arxiv"
do 
    for num_layers in 1 2 4 6 8 12 16
    do 
        echo "Running ${dataset} node classification"
        # run_repeats "task_names ${dataset} num_layers ${num_layers} num_epochs 20 d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK none batch_size 256 emb_dim 128"
        run_srepeats "--dataset ${dataset} --num_layers ${num_layers} --encoder gcn --num_hidden 256 --lr 0.01 --in_drop 0.5"
    done
done 






