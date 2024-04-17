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


slurm_directive="--time=0-3:00:00 --mem=128G --gres=gpu:v100:1 --cpus-per-task=4"
run_repeats "--override ./e2e_graph.yaml"









