## eval normal gcn single
## eval buddy single
## eval seal single
for d in "cora" "citeseer" "pubmed"
do
    # CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets ${d} --model SEALGCN --node_label none
    CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets ${d} --model SEALGCN --hidden_channels 256 --num_hops 3
    CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets ${d} 
done 
## eval normal gcn cotrain
# CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets cora citeseer pubmed --model SEALGCN --node_label none --epochs 50
CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets cora citeseer pubmed --model SEALGCN --epochs 50 --hidden_channels 256 --num_hops 3
CUDA_VISIBLE_DEVICES=2 python3 linkpred.py --pre_train_datasets cora citeseer pubmed --epochs 30


