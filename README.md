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

 


## Code Structures


## Acknowledgements

This code repo is heavily based on [OneForAll](https://github.com/LechengKong/OneForAll), [BUDDY](https://github.com/melifluos/subgraph-sketching), [LLaGA](https://github.com/VITA-Group/LLaGA), [GraphMAE](https://github.com/THUDM/GraphMAE), [Prodigy](https://github.com/snap-stanford/prodigy), [CSTAG](https://github.com/sktsherlock/TAG-Benchmark). Thanks for their sharing!
