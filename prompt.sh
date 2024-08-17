# python3 prompt_pretrain.py --task Edgepred_Gprompt --dataset_name 'arxiv' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 100 --seed 42 --device 0

# python3 prompt_finetune.py --pre_train_model_path './Experiment/pre_trained_model/arxiv/Edgepred_GPPT.GCN.128hidden_dim.pth' --task NodeTask --dataset_name 'cora' --gnn_type 'GCN' --prompt_type 'GPPT' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0

# python3 prompt_finetune.py --pre_train_model_path './Experiment/pre_trained_model/arxiv/Edgepred_GPPT.GCN.128hidden_dim.pth' --task NodeTask --dataset_name bookhis --gnn_type 'GCN' --prompt_type 'GPPT' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0 --normal_split

# python3 prompt_pretrain.py --task Edgepred_Gprompt --dataset_name 'cora' 'citeseer' 'pubmed' 'arxiv' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 10 --seed 42 --device 0
# export OMP_NUM_THREADS=1
for dataset in "cora" "bookhis" "amazonratings"
do 
    echo "GPrompt $dataset"
    python3 prompt_finetune.py --pre_train_model_path /egr/research-dselab/chenzh85/nips/MyOFA/Experiment/pre_trained_model/arxiv2/Edgepred_Gprompt.GCN.128hidden_dim.pth --task NodeTask --dataset_name $dataset --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0 
    python3 prompt_finetune.py --pre_train_model_path /egr/research-dselab/chenzh85/nips/MyOFA/Experiment/pre_trained_model/arxiv2/Edgepred_Gprompt.GCN.128hidden_dim.pth --task NodeTask --dataset_name $dataset --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0 --normal_split

    # echo "Gprompt $dataset"
    # python3 prompt_finetune.py --pre_train_model_path './Experiment/pre_trained_model/co/Edgepred_Gprompt.GCN.128hidden_dim.pth' --task NodeTask --dataset_name $dataset --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0 --normal_split
done 
# for dataset in 'arxiv23' 'arxiv' 'cora' 'citeseer'
# do
#     echo "GPPT $dataset"
#     OMP_NUM_THREADS=1 python3 prompt_finetune.py --pre_train_model_path '/egr/research-dselab/chenzh85/nips/MyOFA/Experiment/pre_trained_model/co/Edgepred_GPPT.GCN.128hidden_dim.pth' --task NodeTask --dataset_name $dataset --gnn_type 'GCN' --prompt_type 'GPPT' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.01 --decay 1e-5 --seed 42 --device 0 --normal_split
#     # echo "Gprompt $dataset"
#     # python3 prompt_finetune.py --pre_train_model_path '/egr/research-dselab/chenzh85/nips/MyOFA/Experiment/pre_trained_model/co/Edgepred_Gprompt.GCN.128hidden_dim.pth' --task NodeTask --dataset_name $dataset --gnn_type 'GCN' --prompt_type 'Gprompt' --shot_num 3 --hid_dim 128 --num_layer 2  --lr 0.02 --decay 2e-6 --seed 42 --device 0 --normal_split
# done

# python3 prompt_pretrain.py --task Edgepred_Gprompt --dataset_name 'cora' 'citeseer' 'arxiv23' 'arxiv' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 20 --seed 42 --device 0

# python3 prompt_pretrain.py --task Edgepred_GPPT --dataset_name 'cora' 'citeseer' 'arxiv23' 'arxiv' --gnn_type 'GCN' --hid_dim 128 --num_layer 2 --epochs 20 --seed 42 --device 0