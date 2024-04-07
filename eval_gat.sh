#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    cfg_overrides=$1
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    name=$2

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python3 sweep.py ${cfg_overrides}"
    common_params="${cfg_overrides}"

    echo "Run program: ${main}"


    script="sbatch ${slurm_directive} -o \"out/gat_${cfg_overrides}\" run/wrapper.sb ${main}"
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

slurm_directive="--time=0-3:30:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
# obj="--dataset citeseer --max_epoch 300 --count 50 --mode sup --encoder gcn" 
# run_repeats "${obj}" "gcn_${d}_sup"
for d in "cora" "citeseer" "pubmed" "arxiv" "arxiv23" "products" "amazonratings" "bookchild" "bookhis" "elecomp" "elephoto" "sportsfit" "wikics"
do
    case "$d" in
        "cora" | "citeseer" | "pubmed")
            max_epoch=500
            ;;
    *)
            max_epoch=1000
            ;;
    esac
    obj="--dataset ${d} --max_epoch ${max_epoch} --count 50 --mode sup" 
    run_repeats "${obj}" "gat_${d}_sup"
    obj="--dataset ${d} --max_epoch ${max_epoch} --count 50 --mode ssl --linear_prob"
    run_repeats "${obj}" "gat_${d}_ssl"
done









