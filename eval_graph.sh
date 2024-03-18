for d in "chemblpre" "chempcba" "chemhiv"
do
CUDA_VISIBLE_DEVICES=7 python run_cdm.py task_names ${d} d_multiple 1 d_min_ratio 1 lr 0.001 num_layers 7 num_epochs 50 dropout 0.15
done

CUDA_VISIBLE_DEVICES=7 python run_cdm.py --override /egr/research-dselab/chenzh85/OneForAll/e2e_graph.yaml 
