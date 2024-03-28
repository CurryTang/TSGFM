#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
	dataset=$1
    device=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python transductive_ssl.py \
	--device $device \
	--dataset $dataset \
	--mask_rate 0.5 \
	--encoder "gat" \
	--decoder "gat" \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_hidden 512 \
	--num_heads 4 \
	--max_epoch 300 \
	--max_epoch_f 200 \
	--lr 0.001 \
	--weight_decay 0 \
	--lr_f 0.01 \
	--weight_decay_f 1e-4 \
	--activation prelu \
	--optimizer adam \
	--drop_edge_rate 0.0 \
	--loss_fn "sce" \
	--seeds 0 1 2 \
	--replace_rate 0.05 \
	--alpha_l 3 \
	--linear_prob \
	--scheduler \
	--use_cfg"
    common_params="${cfg_overrides}"

    echo "Run program: ${main}"


    script="sbatch ${slurm_directive} -J ofa run/wrapper.sb ${main}"
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
# run_repeats "cora 0"
# run_repeats "citeseer 0"
# run_repeats "pubmed 0"
run_repeats "arxiv" "0"
run_repeats "arxiv23" "0"











