CUDA_VISIBLE_DEVICES=1 python3 sweep.py  --pre_train_datasets cora citeseer arxiv arxiv23 --encoder gat --split_mode graphsaint --subgraph_size 1024 --method graphmae --cpuinf & 

CUDA_VISIBLE_DEVICES=2 python3 sweep.py  --pre_train_datasets cora citeseer arxiv arxiv23 --encoder gcn --split_mode graphsaint --subgraph_size 1024 --method dgi --cpuinf & 


CUDA_VISIBLE_DEVICES=3 python3 sweep.py  --pre_train_datasets products bookhis bookchild elephoto elecomp sportsfit --encoder gcn --split_mode graphsaint --subgraph_size 4096 --method dgi & 

CUDA_VISIBLE_DEVICES=4 python3 sweep.py  --pre_train_datasets cora citeseer arxiv arxiv23 products bookhis bookchild elephoto elecomp sportsfit pubmed wikics --encoder gcn --split_mode graphsaint --subgraph_size 4096 --method dgi &