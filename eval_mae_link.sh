# for d in "cora-link" "citeseer-link" "pubmed-link"
# do 
#     CUDA_VISIBLE_DEVICES=3 python3 sweep.py --pre_train_dataset ${d} --method graphmae --subgraph_size -1 --split_mode graphsaint --count 50
# done 

CUDA_VISIBLE_DEVICES=7 python3 sweep.py --pre_train_dataset cora-link citeseer-link pubmed-link --method graphmae --subgraph_size 1024 --split_mode graphsaint --count 50 

CUDA_VISIBLE_DEVICES=7 python3 sweep.py --pre_train_dataset cora-link citeseer-link pubmed-link arxiv-link arxiv23-link --method graphmae --subgraph_size 1024 --split_mode graphsaint --count 50 

CUDA_VISIBLE_DEVICES=7 python3 sweep.py --pre_train_dataset bookhis-link bookchild-link elecomp-link elephoto-link sportsfit-link products-link --method graphmae --subgraph_size 1024 --split_mode graphsaint --count 50 