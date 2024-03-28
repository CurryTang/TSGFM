#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    cfg_overrides=$1
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python run_cdm.py ${cfg_overrides}"
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

slurm_directive="--time=0-2:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=4"
# run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_ctation_oneforall.yaml"
run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_ctation_oneforall.yaml d_multiple 1,1,1,1,1 d_min_ratio 1,1,1,1,1"
# run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_ctation_oneforall.yaml d_multiple 2,1.5,2.5,0.7,1.0 d_min_ratio 1,1,2,0.2,0.3"
# run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_ctation_oneforall.yaml d_multiple 1000,1100,2200,0.2,1.0 d_min_ratio 5,6,10,0.02,0.1"
# run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_ctation_oneforall.yaml d_multiple 3,3,4,1,1.2 d_min_ratio 1,0.7,1,0.1,0.2"
# run_repeats "--override /mnt/home/chenzh85/graphlang/PyGFM/MyOFA/eval_citation_planetoid.yaml d_multiple 1,1,1 d_min_ratio 1,1,1 " 











