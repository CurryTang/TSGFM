import tables
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data_
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM,AutoModelForCausalLM,LlamaTokenizer
import os
import shutil
import joblib
from graphadapter.prompt_config import get_template_by_dataset
from tqdm import tqdm
import sys
import h5py

sys.path.append(os.getcwd())
class RawTextData(data_.Dataset):
    def __init__(self, text,node_id):
        self.text = text
        self.node_id = node_id
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return (self.text[idx],self.node_id[idx])
    
def pretrain_collate_fn(data_tuple):   
    
    seq = [torch.tensor(sq[0]) for sq in data_tuple]
    node_id = [sq[1] for sq in data_tuple]
    seq = pad_sequence(seq, batch_first=True, padding_value=tokenizer.pad_token_id)   
    node_id = torch.tensor(node_id).view(-1,1)
    node_id = node_id.repeat(1,seq.shape[1])
    return seq, node_id

def h5_to_npy_mmap(h5_filename, npy_filename, dataset_name):
    """
    Saves a dataset from an HDF5 file into a NumPy .npy file using memory-mapping.

    Args:
        h5_filename: The name of the HDF5 file.
        npy_filename: The name of the output NumPy .npy file.
        dataset_name: The name of the dataset within the HDF5 file.
    """

    with h5py.File(h5_filename, 'r') as h5_file:
        dataset = h5_file[dataset_name]

        # Create a memory-mapped NumPy array for the output
        npy_mmap = np.memmap(npy_filename, dtype=dataset.dtype, mode='w+', shape=dataset.shape)

        # Copy data from the HDF5 dataset to the memory-mapped array in chunks
        chunk_size = 1024 * 1024  # Adjust chunk size as needed
        for start in tqdm(range(0, dataset.shape[0], chunk_size)):
            end = min(start + chunk_size, dataset.shape[0])
            npy_mmap[start:end] = dataset[start:end]

        # Flush changes to disk and close the memory-mapped array
        npy_mmap.flush()
        del npy_mmap 

def build_pretrain_data_by_tables(model,tokenizer,x_text,save_path,template_l_id,device,args):
    
    template_l_id = tokenizer.encode(template_l)[0:]
    template_l_id = torch.tensor(template_l_id).view(1,-1)
    
    token_embedding_path = save_path+'token_embeddings.h5' 
    f = tables.open_file(token_embedding_path, mode='w')
    atom = tables.Float16Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1280))
    f.close()
    
    sentence_embedding_path = save_path+'sentence_embeddings.h5'
    f = tables.open_file(sentence_embedding_path, mode='w')
    atom = tables.Float16Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1280))
    f.close()
    
    token_node_ids_path = save_path+'token_node_ids.h5'
    f = tables.open_file(token_node_ids_path, mode='w')
    atom = tables.IntAtom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1))
    f.close()
    
    token_label_path = save_path+'token_labels.h5'
    f = tables.open_file(token_label_path, mode='w')
    atom = tables.IntAtom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1))
    f.close()
    
    model.to(device)
    feature_ls=[]
    test_max = 0
    for text in list(x_text):
        feature_ls.append(text)
    print('total node: ', len(feature_ls))
    
    feature_ls_ids = []
    # import ipdb; ipdb.set_trace()
    for f in tqdm(feature_ls):
        feature_ls_ids.append(tokenizer(f,padding=True,truncation=True, max_length=args.max_length)['input_ids'])
    nodedata_ = RawTextData(feature_ls_ids,list(range(len(feature_ls))))
    node_data_loader = DataLoader(nodedata_, batch_size=args.batch_size, shuffle=False,collate_fn=pretrain_collate_fn)
    token_node_ids_ls = []
    labels_ls = []
    embeddings_ls = []
    word_num_ls = []
    cls_embeddings_ls = []

    for i in range(1):
        train_position = []
        for (text,node_id) in tqdm(node_data_loader):
            with torch.no_grad():
                mlm_text_id, labels = text, text[..., 1:].contiguous()
                
                #print(labels)
                mlm_text_id = mlm_text_id[:,1:]
                labels = labels[:,1:]
                node_id = node_id[:,1:]
                
                prompt_l = template_l_id.repeat(mlm_text_id.shape[0],1)#.to(device)
                prompt_labels = torch.zeros_like(prompt_l)
                node_id = torch.cat((prompt_labels-1,node_id),dim=1)
                mlm_text_id = torch.cat((prompt_l,mlm_text_id),dim=1)
                labels = torch.cat((prompt_labels,labels),dim=1)
                
                # import ipdb; ipdb.set_trace()
                
                attention_mask = (mlm_text_id != tokenizer.pad_token_id).long()#.half()
                
                mlm_text_id = mlm_text_id.to(device)
                attention_mask = attention_mask.to(device)
                # import ipdb; ipdb.set_trace()
                embeddings = model.transformer(input_ids=mlm_text_id, attention_mask=attention_mask)[0]
                embedding_dim = embeddings.shape[-1]
                # import ipdb; ipdb.set_trace()
                prompt_last_position = attention_mask.sum(dim=1)-1
                cls_embedding = embeddings.gather(1,prompt_last_position.view(-1,1,1).repeat(1,1,embedding_dim)).view(-1,embedding_dim)
                #cls_embeddings_ls.append(cls_embedding.to('cpu'))
                
                batch_cls_embedding = cls_embedding.to('cpu').numpy()
                
                embeddings = embeddings[:, :-1, :].contiguous()
                node_id = node_id[...,:-1]
                num = (labels!=-0).sum(dim=1)
                token_node_ids = []
                
                
                node_ids = node_id[labels!=0].view(-1,1).to('cpu').numpy()
                
                token_node_ids_ls.append(node_id[labels!=0])
                embeddings = embeddings[labels!=0,:].to('cpu').numpy()
                labels = labels[labels!=0].view(-1,1).to('cpu').numpy()
               
                
                f = tables.open_file(token_embedding_path, mode='a')
                # import ipdb; ipdb.set_trace()
                f.root.data.append(embeddings)
                f.close()

                f = tables.open_file(sentence_embedding_path, mode='a')
                f.root.data.append(batch_cls_embedding)
                f.close()
                
                
                f = tables.open_file(token_node_ids_path, mode='a')
                f.root.data.append(node_ids)
                f.close()
                    
                    
                f = tables.open_file(token_label_path, mode='a')
                f.root.data.append(labels)
                f.close()
    return token_embedding_path,sentence_embedding_path,token_node_ids_path,token_label_path

def convert_tables_to_npy(save_path):
    token_embedding_path = save_path+'token_embeddings.h5' 
    token_node_ids_path = save_path+'token_node_ids.h5'
    token_label_path = save_path+'token_labels.h5'
    sentence_embedding_path = save_path+'sentence_embeddings.h5'

    token_node_ids = tables.open_file(token_node_ids_path, mode='r+').root.data.read()
    np.save(save_path+'token_node_ids.npy',token_node_ids[:,0])
    print("Save token_node_ids.npy")
    
    token_labels = tables.open_file(token_label_path, mode='r+').root.data.read()
    np.save(save_path+'token_labels.npy',token_labels[:,0])
    print("Save token_labels.npy")
   
    token_embeddings = tables.open_file(token_embedding_path, mode='r+').root.data.read()
    print("Open token_embeddings.h5")
    np.save(save_path+'token_embeddings.npy',token_embeddings)
    print("Save token_embeddings.npy")
    
    # h5_to_npy_mmap(token_embedding_path, save_path+'token_embeddings.npy', 'data')
    # print("Save token_embeddings.npy")
    
    sentence_embeddings = tables.open_file(sentence_embedding_path, mode='r+').root.data.read()
    np.save(save_path+'sentence_embeddings.npy',sentence_embeddings)
    print("Save sentence_embeddings.npy")
    
    return True

def cut_input_with_max_length(input_string, max_length):
    """
    Cuts the input string to keep only the first 'max_length' tokens.

    Args:
        input_string: The string to be cut.
        max_length: The maximum number of tokens to keep.

    Returns:
        The cut string containing only the first 'max_length' tokens.
    """

    # Tokenize the input string
    tokens = input_string.split()  # Basic tokenization, adjust as needed

    # Cut the tokens to the desired length
    cut_tokens = tokens[:max_length]

    # Join the cut tokens back into a string
    cut_string = " ".join(cut_tokens)

    return cut_string


def get_prompt_embedding(model,tokenizer,x,template_l,template_r,device,args=None):
    feature_ls=[]
    for text in list(x):
        feature_ls.append(text)
    feature_ls_ids = []
    # import ipdb; ipdb.set_trace()
    for f in feature_ls:
        feature_ls_ids.append(tokenizer(template_l+cut_input_with_max_length(f, args.max_length)+template_r,padding=True,truncation=True, max_length=512)['input_ids'])
    nodedata_ = RawTextData(feature_ls_ids,list(range(len(feature_ls))))
    node_data_loader = DataLoader(nodedata_, batch_size=args.batch_size, shuffle=False,collate_fn=pretrain_collate_fn)
    prompt_embeddings_ls = []
    embedding_dim=model.config.hidden_size
    model = model.to(device)
    for i in range(1):
        train_position = []
        for (text,node_id) in tqdm(node_data_loader):
            with torch.no_grad():
                text_id, labels = text[:,:], text[:, :]
                
                attention_mask = (text_id != tokenizer.pad_token_id).long()
                text_id = text_id.to(device)
                attention_mask = attention_mask.to(device)
                output = model.transformer(input_ids=text_id, attention_mask=attention_mask)[0]
                
                embeddings = output[..., :-1, :].contiguous()
                labels = labels[..., 1:].long()
                
                prompt_last_position = attention_mask.sum(dim=1)-1
                
                prompt_embedding = output.gather(1,prompt_last_position.view(-1,1,1).repeat(1,1,embedding_dim)).view(-1,embedding_dim)
                prompt_embeddings_ls.append(prompt_embedding.to('cpu'))
    prompt_embedding = torch.cat(prompt_embeddings_ls,dim=0)
    prompt_embedding = prompt_embedding.numpy()
    return prompt_embedding

def save_lm_head(model):
    lm_head_path = "./pretrain_models/head/"
    if os.path.exists(lm_head_path):
        shutil.rmtree(lm_head_path, True)    
    os.makedirs(lm_head_path)
    joblib.dump(model.lm_head.to('cpu'),open(f'{lm_head_path}lm_head.pkl','wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('preprocess text-attributed graph by LLMs to gain the token embedding')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='instagram')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of llama 2')
    parser.add_argument('--max_length', type=int, default=2048) 
    # /localscratch/chenzh85/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24
    parser.add_argument('--plm_path', type=str, default='/localscratch/chenzh85/gpt2-large', help='path of llama 2')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--pretrain_save_path', type=str, default='./token_embedding/', help='path of saving pretrain data')
    parser.add_argument('--prompt_save_path', type=str, default='./prompt_embedding/', help='path of saving prompt embedding')
    parser.add_argument('--type',type=str,default='pretrain',help='preprocess type',choices = ['pretrain','prompt','all','convert'])
    args = parser.parse_args()
    
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    device = args.device
    save_path = args.pretrain_save_path+args.dataset_name+'/'

    ##load Llama 2
    if(args.type != 'convert'):
        model = AutoModelForCausalLM.from_pretrained(args.plm_path,low_cpu_mem_usage=True,torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path,use_fast=False)
        tokenizer.pad_token='[PAD]' # for batch preprocess
    
        save_lm_head(model)
        data = torch.load(f'cache_data_minilm/{args.dataset_name}/processed/geometric_data_processed.pt')
        texts = torch.load(f'cache_data_minilm/{args.dataset_name}/processed/texts.pkl')
        for i in range(len(texts)):
            if 'feature node.' in texts[i]:
                texts[i] = texts[i].replace('feature node.','')
        x_text = texts[0]

    if(args.type == 'pretrain') or (args.type=='all'):
        
        if os.path.exists(save_path):
            shutil.rmtree(save_path, True)    
        os.makedirs(save_path)
        
        template_l,template_r = get_template_by_dataset(args.dataset_name)
        print("template_l:",template_l)
        print()
        print("template_r",template_r)
        token_embedding_path,sentence_embedding_path,token_node_ids_path,token_label_path = build_pretrain_data_by_tables(model,tokenizer,x_text,save_path,template_l,args.device,args)
        convert_tables_to_npy(save_path)

    if(args.type == 'convert') or (args.type!='pretrain'): 
        ## if out-of-memory, and the .h5 data have be saved, consider covert-only to transform .h5 to .npy
        convert_tables_to_npy(save_path)

    if(args.type == 'prompt') or (args.type=='all'):
        save_path = args.prompt_save_path+args.dataset_name+'/'
        template_l,template_r = get_template_by_dataset(args.dataset_name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path, True)    
        os.makedirs(save_path)
        prompt_embedding = get_prompt_embedding(model,tokenizer,x_text,template_l,template_r,args.device,args)
        np.save(f'{save_path}/prompt_embedding.npy',prompt_embedding)
    