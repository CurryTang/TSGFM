#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    mode=$1
    cotrain=$2
    cfg_overrides=$3

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python3 sweep.py --pre_train_datasets ${cotrain} ${cfg_overrides}"
    common_params="${cfg_overrides}"

    echo "Run program: ${main}"


    script="sbatch ${slurm_directive} run/wrapper.sb ${main}"
    echo $script
    eval $script &> cotrain.log 2>&1
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

slurm_directive="--time=0-3:59:00 --mem=160G --cpus-per-task=6"



run_repeats "graphmae" "products bookhis bookchild elephoto elecomp sportsfit" " --max_epoch 15 --warmup --split_mode graphsaint --subgraph_size 4096 --batch_size 16 --device -1 --count 30"
 

