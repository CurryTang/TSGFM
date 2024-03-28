#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    mode=$1
    cotrain=$2
    cfg_overrides=$3

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python3 ${mode}.py --pre_train_datasets ${cotrain} ${cfg_overrides}"
    common_params="${cfg_overrides}"

    echo "Run program: ${main}"


    script="sbatch ${slurm_directive} run/wrapper.sb ${main}"
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

slurm_directive="--time=0-1:30:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"

## ssl
# run_repeats "self-supervised" "cora citeseer pubmed arxiv arxiv23" ""

## ssl linear probe
run_repeats "self-supervised" "cora citeseer pubmed arxiv arxiv23" "--linear_prob --max_epoch 200"

# ## sup gcn
# run_repeats "supervised" "cora citeseer pubmed arxiv arxiv23" "--encoder gcn --multi_sup_mode"

# ## sup gat
# run_repeats "supervised" "cora citeseer pubmed arxiv arxiv23"  "--multi_sup_mode"

# ## sup gat small
# run_repeats "supervised" "cora citeseer pubmed" "--encoder gcn" "--multi_sup_mode"

## ssl gat small
# run_repeats "self-supervised" "cora citeseer pubmed" "--max_epoch 200"
## ssl gat small linear probe
run_repeats "self-supervised" "cora citeseer pubmed" "--linear_prob --max_epoch 200 --initial_weight 1. 1. 1. --min_ratio 1. 1. 1."