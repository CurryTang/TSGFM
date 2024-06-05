# ALL GNN related experiments are here!!
function run_repeats {
    dscap=$1 # default: 50010 20000 -> 12 hours
    shot=$2 # default: 3, change this for downstream tasks
    task=$3 # default: cls_nm_sb
    prefix=$4 #default: MAG_PT_PRODIGY
    embdim=$5 #default: 256
    lr=$6 #default: 3e-4
    data=$7 #default: mag240m
    way=$8 #default: 15
    feat=$9

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python experiments/run_single_experiment.py --dataset ${data}  --root /mnt/home/chenzh85/graphlang/PyGFM/datasets --original_features ${feat} -ds_cap ${dscap} -val_cap 100 -test_cap 100 --emb_dim ${embdim} --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr ${lr} -way ${way} -shot ${shot} -qry 4 -eval_step 5000 -task ${task}  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix ${prefix}"
    # out_dir="results/${dataset}"  # <-- Set the output dir.
    # common_params="out_dir ${out_dir} ${cfg_overrides} wandb.use False"
    # python experiments/run_single_experiment.py --dataset mag240m --root <DATA_ROOT> --original_features True --input_dim 768 --emb_dim 256 -ds_cap 10010 -val_cap 100 -test_cap 100 --epochs 1 -ckpt_step 1000 -layers S2,U,A -lr 1e-3 -way 30 -shot 1 -qry 4 -eval_step 500 -task same_graph  -bs 1 -aug ND0.5,NZ0.5 -aug_test True --device 0 --prefix MAG_Contrastive

    echo "Run program: ${main}"
    # echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    script="sbatch ${slurm_directive} wrapper.sb ${main}"
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


## Experiments for full-version prodigy

## data scaling
# run_repeats 24000 3 cls_nm_sb MAG_PT_PRODIGY 32 3e-4
# run_repeats 24000 3 cls_nm_sb MAG_PT_PRODIGY 64 3e-4 
# run_repeats 24000 3 cls_nm_sb MAG_PT_PRODIGY 128 3e-4
slurm_directive="--time=0-8:00:00 --mem=128G --gres=gpu:a100:1 --cpus-per-task=8"
run_repeats "24000" "3" "neighbor_matching" "MAG_PT_PRODIGY" "256" "3e-4" "arxiv" "30" "False"
run_repeats "24000" "3" "cls_nm_sb" "MAG_PT_PRODIGY" "256" "3e-4" "arxiv" "30" "False"
# run_repeats 24000 3 cls_nm_sb MAG_PT_PRODIGY 512 3e-4


slurm_directive="--time=0-8:00:00 --mem=256G --gres=gpu:a100:1 --cpus-per-task=8"
# ## Experiments for prodigy with only supervised signals
# # run_repeats 24000 3 neighbor_matching MAG_PG_NM 32 3e-4
# # run_repeats 24000 3 neighbor_matching MAG_PG_NM 64 3e-4
# run_repeats "24000" "3" "neighbor_matching" "MAG_PG_NM" "128" "3e-4" "mag240m" "30" "True"
# run_repeats 24000 3 neighbor_matching MAG_PG_NM 256 3e-4
# run_repeats 24000 3 neighbor_matching MAG_PG_NM 512 3e-4



## Experiments for prodigy with only self-supervised signals
# run_repeats 24000 3 classification MAG_PG_MT 32 3e-4
# run_repeats 24000 3 classification MAG_PG_MT 64 3e-4
# run_repeats 24000 3 classification MAG_PG_MT 128 3e-4
# run_repeats 24000 3 classification MAG_PG_MT 256 3e-4
# run_repeats 24000 3 classification MAG_PG_MT 512 3e-4

## Experiments for contrastive learning
# run_repeats 24000 3 same_graph MAG_PG_CL 32 1e-3
# run_repeats 24000 3 same_graph MAG_PG_CL 64 1e-3
# run_repeats 24000 3 same_graph MAG_PG_CL 128 1e-3
# run_repeats 24000 3 same_graph MAG_PG_CL 256 1e-3
# run_repeats 24000 3 same_graph MAG_PG_CL 512 1e-3