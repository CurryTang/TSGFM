for d in "dblp_node"
do
    python3 run_cdm.py task_names $d rwpe 20 num_epochs 30 exp_name $d_rwpe
done

python3 run_cdm.py task_names "cora_node,citeseer_node,dblp_node,arxiv,arxiv23" rwpe 20 num_epochs 30 exp_name "cotrain_citation_node_rwpe" d_multiple 1,1,1,1,1 d_min_ratio 1,1,1,1,1

python3 run_cdm.py task_names "cora_node,citeseer_node,dblp_node,arxiv,arxiv23" num_epochs 30 exp_name "cotrain_citation_node_rescale" d_multiple 10,10,10,1,1 d_min_ratio 1,1,1,1,1