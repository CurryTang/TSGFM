## Single-task graph-level
# for d in "cora_node" "citeseer_node" "pubmed_node" "arxiv"
# do
# CUDA_VISIBLE_DEVICES=3 python run_cdm.py task_names ${d} d_multiple 1 d_min_ratio 1 lr 0.001 num_layers 3 num_epochs 20 dropout 0.15
# done

echo "Test Cora node classification"
python run_cdm.py task_names cora_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64
python run_cdm.py task_names cora_node num_layers 3 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64

echo "Test Citeseer node classification"
python run_cdm.py task_names citeseer_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64
python run_cdm.py task_names citeseer_node num_layers 3 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64

echo "Test Pubmed node classification"
python run_cdm.py task_names pubmed_node num_layers 5 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64
python run_cdm.py task_names pubmed_node num_layers 3 num_epochs 20 d_multiple 10.0 d_min_ratio 10.0 lr 0.0001 JK none batch_size 64

echo "Test Arxiv node classification"
python run_cdm.py task_names arxiv num_layers 5 num_epochs 20 d_multiple 1 d_min_ratio 1 lr 0.0001 JK none batch_size 64
python run_cdm.py task_names arxiv num_layers 3 num_epochs 20 d_multiple 1 d_min_ratio 1 lr 0.0001 JK none batch_size 64

echo "Test multiple datasets"

CUDA_VISIBLE_DEVICES=3 python run_cdm.py --override /egr/research-dselab/chenzh85/OneForAll/e2e_all_citation_node.yaml 
CUDA_VISIBLE_DEVICES=3 python run_cdm.py --override /egr/research-dselab/chenzh85/OneForAll/e2e_all_citation_node_3.yaml 

