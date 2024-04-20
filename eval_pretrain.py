import sys
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import os.path as osp
from graphllm.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from graphllm.converstation import conv_templates, SeparatorStyle
from graphllm.builder import load_pretrained_model
from graphllm.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from graphllm.utils import classification_prompt, link_prediction_prompt
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(hop+1)]
    return pretrained_embs


def set_label_names(data, label_csv_path):
    label_pd = pd.read_csv(label_csv_path)
    if hasattr(data, 'label_names'):
        return data 
    label_names = label_pd['name'].tolist()
    data.label_names = label_names
    return data

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # data_dir=os.path.expanduser(args.data_dir)
    data_path = osp.join(args.data_saved_path, args.dataset, 'processed', "geometric_data_processed.pt")
    data_dir = osp.join(args.data_saved_path, args.dataset, 'processed')
    if args.task in  ["nc"]:
        prompt_file = os.path.join(data_dir, f"sampled_2_10_test.jsonl")
    elif args.task in ["lp"]:
        prompt_file = os.path.join(data_dir, f"edge_sampled_2_10_only_test.jsonl")
    else:
        raise ValueError

    data = torch.load(data_path)[0]
    if hasattr(data, "train_masks"):
        data.train_mask = data.train_masks[0]
        data.val_mask = data.val_masks[0]
        data.test_mask = data.test_masks[0]
    
        del data.train_masks
        del data.val_masks
        del data.test_masks
    elif dataset == 'arxiv':
        arxiv_mask = torch.load(osp.join(data_args.data_saved_path, dataset, 'processed', 'arxiv_mask.pt'))
        data.train_mask = arxiv_mask['train']
        data.val_mask = arxiv_mask['valid']
        data.test_mask = arxiv_mask['test']
    set_label_names(data, osp.join(data_args.data_saved_path, dataset, 'processed','categories.csv'))
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    n = data.num_nodes
    if args.task == "lp":
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop)[0]
    else:
        mask = torch.full([n], fill_value=False, dtype=torch.bool)
        for q in questions:
            idx = q["id"]
            if "lp" in  args.task:
                assert len(idx) == 2
                mask[idx[0]] = True
                mask[idx[1]] = True
            elif args.task  in ["nc", "nd", "nctext"]:
                assert isinstance(idx, int)
                mask[idx] = True
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop)
        index = torch.full([n], fill_value=n + 1, dtype=torch.long)
        test_index = torch.arange(mask.sum())
        index[mask] = test_index
    structure_emb = None

    for line in tqdm(questions):
        idx = line["id"]
        if args.task == 'nc':
            human_conv, gpt_conv = classification_prompt(args.category, label_names = data.label_names, gt=data.y[idx])
        else:
            prompt, human_conv, gpt_conv = link_prediction_prompt(line['gt'])
        ## dynamically generate the prompts
        line['conversations'] = []
        line['conversations'].append(human_conv)
        line['conversations'].append(gpt_conv)
        line['conversations'][0]['value'] = prompt
        qs = line["conversations"][0]['value']
        cur_prompt = qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        if args.task == "lp":
            mp = MP()
            center_nodes = []
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id] * (args.use_hop + 1)
                center_nodes.append(center_id)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_embs = [pretrained_emb[center_id].cuda()]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index,
                                                                    relabel_nodes=True)
            local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
                        (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
            edge_index = edge_index[:, ~local_edge_mask]
            local_x = pretrained_emb[subset].cuda()
            n = subset.shape[0]
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index)
            edge_index = edge_index.cuda()
            row, col = edge_index
            deg = degree(col, n, dtype=pretrained_emb.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # local_x = pretrained_emb
            for _ in range(args.use_hop):
                local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                graph_embs.append(local_x[mapping])
            graph_emb = torch.stack(graph_embs, dim=1)
        else:
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id]*(args.use_hop+1)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if stop_str != '' and outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            raise ValueError(e)
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "graph": line['graph'],
                                   "text": outputs,
                                   "gt":line["conversations"][1]['value'],
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_saved_path", type=str, default="cache_data_minilm")
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="HO")
    parser.add_argument("--category", type=str, default="paper")
    args = parser.parse_args()

    eval_model(args)