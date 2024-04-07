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


for dataset in "toxcast"
do 
    if [ "$dataset" == "chempcba" ]; then
        time=8
        epoch=75
    else
        time=3
        epoch=75
    fi
    slurm_directive="--time=0-${time}:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
    run_repeats "task_names ${dataset} num_layers 7 num_epochs ${epoch} d_multiple 1.0 d_min_ratio 1.0 lr 0.0001 JK none batch_size 512 emb_dim 128"
done 