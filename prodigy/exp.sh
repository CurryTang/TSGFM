# ALL GNN related experiments are here!!
function run_repeats {
    start=$1
    end=$2

    main="singularity exec --nv /mnt/home/chenzh85/pytorch.sif python experiments/split_mag240m.py --start ${start} --end ${end}"

    echo "Run program: ${main}"
    # echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    script="sbatch wrapper.sb ${main}"
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

# run_repeats 0 20

for i in $(seq 0 100 900); do 
  echo "Current value: $i"
  run_repeats $i $(($i + 100))
done

