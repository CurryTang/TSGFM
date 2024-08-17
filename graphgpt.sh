dsname=amazonratings
output_model=/localscratch/chenzh85/GraphGPT-7B-mix-all 
datapath=instruct_ds/${dsname}_test_instruct_GLBench.json
graph_data_path=instruct_ds/${dsname}.pt
res_path=./GLBench_${dsname}_nc_output

num_gpus=1

python graphgpt/eval/run_graphgpt.py --dataset ${dsname} --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --num_gpus ${num_gpus}