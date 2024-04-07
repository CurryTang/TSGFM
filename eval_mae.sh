#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
	dataset=$1
    device=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python transductive_ssl.py \
	--device 0 \
	--dataset $dataset \
	--mask_method "random" \
    --remask_method "fixed" \
	--mask_rate 0.5 \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_dec_layers 1 \
	--num_hidden 256 \
	--num_heads 4 \
	--num_out_heads 1 \
	--encoder "gat" \
	--decoder "gat" \
	--max_epoch 1000 \
	--max_epoch_f 300 \
	--lr 0.001 \
	--weight_decay 0.04 \
	--lr_f 0.005 \
	--weight_decay_f 1e-4 \
	--activation "prelu" \
	--loss_fn "sce" \
	--alpha_l 3 \
	--scheduler \
	--seeds 0 \
	--lam 0.5 \
	--linear_prob \
	"    
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











