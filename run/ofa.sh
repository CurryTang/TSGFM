#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    cfg_overrides=$1
    cfg_name=$2


    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python run_cdm.py ${cfg_overrides}"

    echo "Run program: ${main}"

    script="sbatch ${slurm_directive} -J ${cfg_name} run/wrapper.sb ${main}"
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


## single dataset



