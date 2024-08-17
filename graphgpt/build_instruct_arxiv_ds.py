import json
import random
import re
import pandas as pd
from tqdm import tqdm
import torch as th
from torch_geometric.utils import subgraph
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils import from_scipy_sparse_matrix
import logging
import copy
import os
from torch_sparse import SparseTensor
from transformers import AutoModel, AutoTokenizer


th.manual_seed(0)
random.seed(0) 
'''
instruct dataset: 
[{'id': 'dsname_train_nodeidx', 'graph': [edge_row, edge_col], 'conversations': [{'from': 'human', 'value': 'human prompting.\n<graph>'}, {'from': 'gpt', 'value': 'gpt response'}]}, {...}]

graph_token: <graph>
'''
dsname = ['amazonratings']
split_type = 'test'

instruct_ds = []

bert = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2", cache_dir = "\
                                 /localscratch/chenzh85")
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2", cache_dir = "/localscratch/chenzh85")

def generate_embedding(sentence, tokenizer, model):
    # Tokenize the sentence and add special tokens
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate embeddings
    with th.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings (we'll use the [CLS] token embedding)
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings[0] 

def generate_label_description(label_df):
    output = ""
    for idx, row in label_df.iterrows():
        output += f"{row['name']}: {row['description']},"
    label_names = label_df['name'].tolist()
    return output, label_names

for d in dsname:
    graph_data = th.load(f"./cache_data_minilm/{d}/processed/geometric_data_processed.pt")[0]
    graph_texts = th.load(f"./cache_data_minilm/{d}/processed/texts.pkl")[0]
    label_df = pd.read_csv(f"./cache_data_minilm/{d}/processed/categories.csv")
    graph_data.raw_texts = graph_texts
    label_exp, label_names = generate_label_description(label_df)
    graph_data.label_name = label_names
    # embeddings = []
    # for text in tqdm(graph_texts):
    #     embeddings.append(generate_embedding(text, tokenizer, bert))
    # graph_data.x = th.stack(embeddings)
    graph_data.x = graph_data.node_text_feat
    
    # graph_data = th.load(f'../../../datasets/{dsname}.pt'
    # if dsname in ['pubmed', 'cora', 'citeseer']:
    #     graph_data.test_mask = graph_data.test_mask[0]
    # import ipdb; ipdb.set_trace()
    if hasattr(graph_data, 'test_mask') and graph_data.test_mask is not None:
        if isinstance(graph_data.test_mask, list):
            graph_data.test_mask = graph_data.test_mask[0]
    else:
        graph_data.test_mask = graph_data.test_masks[0]
    indices = th.nonzero(graph_data.test_mask).reshape(-1)
    select_idx = indices.tolist()
    # select_idx = [0, 1]
    print(len(select_idx))

    print(graph_data.edge_index)
    row, col = graph_data.edge_index
    # import ipdb; ipdb.set_trace()
    sparse_tensor = SparseTensor(row=row, col=col, sparse_sizes=( graph_data.x.shape[0],  graph_data.x.shape[0]))
    s = sparse_tensor.to_scipy() 

    edge_index, edge_attr = from_scipy_sparse_matrix(s) 
    print(f'is undirected: {is_undirected(edge_index)}')
    pyg_data = copy.deepcopy(graph_data)
    pyg_data.num_node = graph_data.x.shape[0]
    del pyg_data.label_name
    del pyg_data.category_names
    del pyg_data.node_text_feat
    del pyg_data.edge_text_feat
    del pyg_data.noi_node_text_feat
    del pyg_data.class_node_text_feat
    del pyg_data.prompt_edge_text_feat
    del pyg_data.one_shot_train, pyg_data.three_shot_train, pyg_data.five_shot_train, pyg_data.one_shot_val, pyg_data.three_shot_val, pyg_data.five_shot_val,pyg_data.one_shot_test, pyg_data.three_shot_test, pyg_data.five_shot_test
    # Data(num_nodes=169343, x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243], train_mask=[169343], val_mask=[169343], test_mask=[169343], edge_index=[169343, 169343, nnz=2315598])
    del pyg_data.raw_texts
    del pyg_data.raw_text
    del pyg_data.label_names 
    del pyg_data.train_masks
    del pyg_data.val_masks
    del pyg_data.test_masks
    del pyg_data.test_mask
    del pyg_data.num_node
    pyg_data.edge_index = pyg_data.edge_index.contiguous()
    pyg_data.x = pyg_data.x.contiguous()
    pyg_data.y = pyg_data.y.contiguous()


    for nidx in tqdm(select_idx): 
        center_node = nidx 
        num_hops = 2
        num_neighbors = 10

        # 邻居采样    
        # import ipdb; ipdb.set_trace()
        sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]),
                                num_neighbors=[num_neighbors] * num_hops, 
                                batch_size=1)

        # 获取子图    
        sampled_data = next(iter(sampler))
        # for sampled_data in sampler:

        # try:
            # if cal_acc(instruct_list[nidx]['output'], instruct_list[nidx]['instruction'], nidx, topk=2) is False: 
        temp_dict = {}
        temp_dict['id'] = f'{d}_{split_type}_{nidx}'
        temp_dict['graph'] = {'node_idx':nidx, 'edge_index': sampled_data.edge_index.tolist(), 'node_list': sampled_data.n_id.tolist()}
        conv_list = []
        conv_temp = {}
        conv_temp['from'] = 'human'
        if d == 'cora':
            conv_temp['value'] = "Given a citation graph: \n<graph>\n \where the 0th node is the target paper, with the following information: \n" +  graph_data.raw_texts[nidx] + "\n" + "\nQuestion: Which of the following category does this paper belong to: " + ",".join(graph_data.label_name) + '. Directly give the full name of the most likely category.'
        elif d == 'bookhis':
            conv_temp['value'] = "Given a product graph: \n<graph>\n \where the 0th node is the target product, with the following information: \n" +  graph_data.raw_texts[nidx] + "\n" + "\nQuestion: Which of the following category does this product belong to: " + ",".join(graph_data.label_name) + '. Directly give the full name of the most likely category.' 
        elif d == 'amazonratings':
            conv_temp['value'] = "Given a product graph: \n<graph>\n \where the 0th node is the target product, with the following information: \n" +  graph_data.raw_texts[nidx] + "\n" + "\nQuestion: Which of the following category does this product belong to: " + ",".join(graph_data.label_name) + '. Directly give the full name of the most likely category.' 
        # conv_temp['value'] = 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n' + graph_data.raw_texts[nidx] + '\n Question: Which of the following specific aspect of diabetes research does this paper belong to: ' + ', '.join(graph_data.label_name) + '. Directly give the full name of the most likely category of this paper.'    
        conv_list.append(copy.deepcopy(conv_temp))
        conv_temp['from'] = 'gpt'
        conv_temp['value'] = 'None'
        conv_list.append(copy.deepcopy(conv_temp))

        temp_dict['conversations'] = conv_list

        instruct_ds.append(temp_dict)
        #import ipdb; ipdb.set_trace()
        # except Exception as e:
        #     logger.info(e)
            


    print(f'total item: {len(instruct_ds)}')
    st_dir = './instruct_ds'
    if not os.path.exists(st_dir):
        os.makedirs(st_dir)
    with open(f'./instruct_ds/{d}_{split_type}_instruct_GLBench.json', 'w') as f:
        json.dump(instruct_ds, f)
    
    # th.save(graph_data, f'./instruct_ds/{d}.pt')

