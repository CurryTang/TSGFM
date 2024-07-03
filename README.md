# Code and Datasets for *Text-space Graph Foundation Models: Comprehensive Benchmarks and New Insights*

This is the code repo accompanying our paper "Text-space Graph Foundation Models: Comprehensive Benchmarks and New Insights."

We implement the following graph foundation model building blocks.

* Graph prompt models (OneForAll, Prodigy)
* GraphLLM (LLaGA)
* Graph Self-supervised learning (GraphMAE, BGRL, DGI, and so on)
* Link prediction-specific models, including BUDDY and SEAL

We support the following two scenarios.

* Co-training: Pre-training on a set of datasets and testing on the same ones
* Pre-training: Pre-training on a set of datasets and testing on unseen ones

## Install

``` 
pip install -r requirements.txt
```

## Datasets

We follow OneForAll's way of managing the datasets. We support the following datasets. 
| Name           | #Graphs | #Nodes | #Edges   | Domains      | Tasks      | #classes |
|----------------|---------|--------|----------|--------------|------------|----------|
| Cora           | 1       | 2708   | 10556    | CS Citation  | Node, Link | 7        |
| CiteSeer       | 1       | 3186   | 8450     | CS Citation  | Node, Link | 6        |
| Arxiv          | 1       | 169343 | 2315598  | CS Citation  | Node, Link | 40       |
| Arxiv23        | 1       | 46198  | 77726    | CS Citation  | Node, Link | 40       |
| History        | 1       | 41551  | 503180   | E-commerce   | Node, Link | 12       |
| Child          | 1       | 76875  | 2325044  | E-commerce   | Node, Link | 24       |
| Computers      | 1       | 87229  | 1256548  | E-commerce   | Node, Link | 10       |
| Photo          | 1       | 48362  | 873782   | E-commerce   | Node, Link | 12       |
| Sportsfit      | 1       | 173055 | 3020134  | E-commerce   | Node, Link | 13       |
| Products       | 1       | 316513 | 19337722 | E-commerce   | Node, Link | 39       |
| Amazon Ratings | 1       | 24492  | 186100   | E-commerce   | Node, Link | 5        |
| Pubmed         | 1       | 19717  | 88648    | Bio Citation | Node, Link | 3        |
| WikiCS         | 1       | 11701  | 431726   | Knowledge    | Node, Link | 10       |
| Tolokers       | 1       | 11758  | 1038000  | Anomaly      | Node, Link | 2        |
| DBLP           | 1       | 14376  | 431326   | CS Citation  | Node, Link | 4        |
| CheMBL         | 365065  | 26     | 112      | Biology      | Graph      | 1048     |
| PCBA           | 437092  | 26     | 56       | Biology      | Graph      | 128      |
| HIV            | 41127   | 26     | 55       | Biology      | Graph      | 2        |
| Tox21          | 7831    | 19     | 39       | Biology      | Graph      | 12       |
| Bace           | 1513    | 34     | 74       | Biology      | Graph      | 2        |
| Bbbp           | 2039    | 24     | 52       | Biology      | Graph      | 2        |
| Muv            | 93087   | 24     | 53       | Biology      | Graph      | 17       |
| Toxcast        | 8575    | 19     | 39       | Biology      | Graph      | 588      |

The processed file versions can be achieved from the following link.

Structures of the processed files:
* `cache_data_{llm encoder name}` (for example, minilm)
  * `dataset_name`
    * `processed`
      * `data.pt`
      * `geometric_data_processed.pt`
      * `pre_filter.pt`
      * `pre_transform.pt`
      * `texts.pkl`

`geometric_data_processed.pt` is the core storage object, and `node_text_feat` stores the processed node features.
`data.pt` contains the index file used to query the attributes stored in `geometric_data_processed.pt`.
A comprehensive introduction of each column can be found in OneForAll's repo. 

To prepare the data, it's okay to generate all raw files yourself (run oneforall for 1 epoch, including all datasets). I recommend you use the [preprocessed files](https://huggingface.co/datasets/zkchen/tsgfm/blob/main/minilmdata.zip) directly and unzip them to the main directory. 


## Code Structures

### Directories

* `configs`: Directory for setting the task/dataset for OneForAll. Add new datasets here
* `data`: data utility files/generation files using the OneForAll data interface
* `gp`: graph utility files from the original OneForAll repo
* `graphllm`: utility files for LLaGA
* `graphmae`: utility files for graphmae
* `link`: utility files for BUDDY
* `models`: model implementations
* `prodigy`: prodigy files
* `subgcon`: utility files/data files for self-supervised learning

### Main entries

* `eval_pretrain_*, eval_res`: main files for LLaGA
* `fulllink.py`: main files for GCN link prediction
* `linkpred.py`: main files for BUDDY/SEAL
* `run_cdm`: main files for OFA
* `sslmain`: main files for SSL
* `simplerlr`: main files for simpleSBERT





## Reproduce the results

### OneForAll
* Co-training setting: just set up a config file similar to `demo/e2e_all_config.yaml`
* Pre-training setting: when loading the pre-trained model, use `gnn_load_path`.

### LLaGA
1. Use `llm_train.sh` to generate checkpoints
2. Use `llm_eval.sh` or `llm_eval_link.sh` to generate the answer files for node/link-level tasks. For example, `bash llm_eval.sh citeseer nc ./checkpoints/llaga-mistral-7b-hf-sbert-4-hop-token-linear-cora.3-citeseer.4-pubmed.3-nc-lp-projector/ citationcross`
3. Use `llmres.sh` to calculate the results

### GCN-link

```
python3 fulllink.py --pre_train_datasets "cora-link" "citeseer-link" "pubmed-link" "arxiv-link" "arxiv23-link" "bookhis-link" "bookchild-link" "sportsfit-link" "products-link" "elecomp-link" "elephoto-link" --encoder gcn --num_layers 3 --num_hidden 128 --batch_size 512
```

### BUDDY/SEAL

```
python3 linkpred.py --pre_train_datasets cora citeseer arxiv arxiv23 bookhis bookchild elecomp elephoto sportsfit products pubmed wikics --model BUDDY --cache_subgraph_features --max_hash_hops 3 --epochs 50
```

```
python3 linkpred.py --pre_train_datasets cora --model SEALGCN --hidden_channels 256 --num_hops 3
```

### SSL

Check the best hyper-parameter in the paper (use cpuinf can do full-batch inference on CPU, which is faster on our environment)
```
python3 sslmain.py --pre_train_datasets arxiv sportsfit products --method graphmae --num_heads 4 --num_out_heads 1 --num_layers 3 --num_hidden 1024 --residual --in_drop 0.5 --attn_drop 0.5 --norm 'batchnorm' --lr 0.01 --weight_decay 1e-5 --activation 'prelu' --mask_rate 0.75 --drop_edge_rate 0 --replace_rate 0.2 --scheduler --lrtype 'cosine' --save_model --max_epoch 5 --subgraph_size 1024 --warmup --cpuinf
```

### Prodigy

pretrain on arxiv
```
python experiments/run_single_experiment.py --dataset arxiv --root <root> --original_features False -ds_cap 24000 -val_cap 100 -test_cap 100 --emb_dim 256 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 30 -shot 3 -qry 4 -eval_step 5000 -task cls_nm_sb -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix MAG_PT_PRODIGY
```

test on History
```
python3 experiments/run_single_experiment.py --dataset bookhis --original_features True -ds_cap 300 -val_cap 300 -test_cap 300 --emb_dim 256 --epochs 1 -ckpt_step 1000 -layers S2,U,M -lr 3e-4 -way 12 -shot 3 -qry 4 -eval_step 50 -task cls_nm_sb  -bs 1 -aug ND0.5,NZ0.5 -aug_test True -attr 1000 --device 0 --prefix test --root <root> -pretrained <ckpt>
```


## Acknowledgements

This code repo is heavily based on [OneForAll](https://github.com/LechengKong/OneForAll)(✨), [BUDDY](https://github.com/melifluos/subgraph-sketching), [LLaGA](https://github.com/VITA-Group/LLaGA), [GraphMAE](https://github.com/THUDM/GraphMAE), [Prodigy](https://github.com/snap-stanford/prodigy), [CSTAG](https://github.com/sktsherlock/TAG-Benchmark). Thanks for their sharing!
