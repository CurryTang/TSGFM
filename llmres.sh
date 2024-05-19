
dataset=${1:-"cora"}
task=${2:-"nc"}
output_path=${3:-"./checkpoints/cora_nc"}


python3 eval_res.py --dataset ${dataset} --task ${task}  --res_path ${output_path}