import torch
import random
from gtext.llm.bedrockm import Bedrock
from tqdm import tqdm
from collections import OrderedDict
from gtext.gllmutils import get_sampled_nodes, num_tokens_from_messages, load_mapping
import numpy as np
import os.path as osp
import ast
import editdistance
from gtext.few_shot_samples import few_shot
import os
# or import cPickle as pickle

def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def top1_label_getter(pred_texts, label_names):
    preds = []
    label_names = [l.lower() for l in label_names]
    for i, t in enumerate(pred_texts):
        match = False
        clean_t = t.replace('.', ' ')
        clean_t = clean_t.lower()
        try:
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            result = ast.literal_eval(list_str)
            res = result[0]
            if res in label_names:
                this = label_names.index(res)
                preds.append(this)
                match = True
            else:
                edits = np.array([editdistance.eval(res, l) for l in label_names])
                this = np.argmin(edits)
                preds.append(this)
                match = True
        except Exception:
            for i, l in enumerate(label_names):
                if l.lower() in clean_t:
                    preds.append(i)
                    match = True
                    break
        if not match:
            edits = np.array([editdistance.eval(clean_t, l) for l in label_names])
            this = np.argmin(edits)
            preds.append(this)

    preds = torch.LongTensor(preds)
    return preds





def prompt_zero_shot(data_obj, sampled_test_node_idxs, train_node_idxs, topk = True, need_class = True, instruction_format = 'arxiv cs xx', mapping = None, all_possible = False, cot = False, memory_limit = 1000000):
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    nl_labels = [data_obj.label_names[i] for i in data_obj.y.numpy()]
    raw_texts = data_obj.raw_texts
    if "arxiv" in instruction_format:
        label_names = [transform_category(x) for x in label_names]
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    data_y = data_obj.y.numpy()
    prompts = []
    selected_raw_texts = [raw_texts[i] for i in sampled_test_node_idxs]
    selected_y = data_y[sampled_test_node_idxs]
    selected_category = [nl_labels[i] for i in sampled_test_node_idxs]
    for t in selected_raw_texts:
        prompt = "Product Description:\n {}\n".format(t)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']"
        prompts.append(prompt)
    if mapping != None:
        return prompts, selected_y, selected_category, human_label_names
    else:
        return prompts, selected_y, selected_category





        

def topk_accuracy(pred_texts, gt, label_names, topk = True, need_clean = True):
    preds = []
    correct = 0
    miss = 0
    label_names = [x.lower() for x in label_names]
    for i, t in enumerate(pred_texts):
        if need_clean:
            clean_t = t.replace('.', ' ')
            clean_t = clean_t.lower()
            clean_t = clean_t.replace('\\', '')
            clean_t = clean_t.replace('_', ' ')
        else:
            clean_t = t
        # import ipdb; ipdb.set_trace()
        try:
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
            result = ast.literal_eval(list_str)

            # import ipdb; ipdb.set_trace()
            res = result[0]
            if res in label_names:
                this = label_names.index(res)
                if this == gt[i]:
                    correct += 1
                    continue
            else:
                miss += 1
                edits = np.array([editdistance.eval(res, l) for l in label_names])
                this = np.argmin(edits)
                if this == gt[i]:
                    correct += 1
                    continue
            
        except Exception:
            miss += 1
            for k, l in enumerate(label_names):
                if l.lower() in clean_t:
                    if k == gt[i]:
                        correct += 1
                    break
    print(miss)
    return correct / len(pred_texts)


def prompt_few_shot(data_obj, few_shot_samples, sampled_test_node_idxs, train_node_idxs, topk = True, need_class = True, instruction_format = 'arxiv cs xx', mapping = None, cot = False, dataset_name = "cora", shots = 3, memory_limit = 1000000):
    nu_labels = data_obj.y.numpy()
    label_names = data_obj.label_names
    nl_labels = [data_obj.label_names[i] for i in data_obj.y.numpy()]
    raw_texts = data_obj.raw_texts
    if mapping != None:
        human_label_names = [mapping[key] for key in data_obj.label_names]
    data_y = data_obj.y.numpy()
    prompts = []
    selected_raw_texts = [raw_texts[i] for i in sampled_test_node_idxs]
    selected_y = data_y[sampled_test_node_idxs]
    selected_category = [nl_labels[i] for i in sampled_test_node_idxs]
    
    few_shot = few_shot_samples["top1"][dataset_name]
    for t in selected_raw_texts:
        prompt = "\n".join(few_shot[:shots])
        if dataset_name == 'amazonratings':
            prompt += """
                name,description
                5 score, "5 score awesome ratings users are extremely satisfied with the products."
                4.5 score, "4.5 score good ratings users are satisfied with the products but there's space to be even better."
                4 score, "4 score good ratings users like the products but there's still much space to be better. "
                3.5 score, "3.5 score average ratings users are neutral about the products."
                0-3 score,"0-3 score bad ratings users think the products are bad."
                Name of the products: My Body Is Private (Albert Whitman Prairie Books)
            """
        prompt += "Product Description:\n {}\n".format(t)
        if need_class:
            if mapping != None:
                prompt += "Task: \n"
                prompt += "There are following categories: \n"
                prompt += (str(human_label_names) + "\n")
            else:
                prompt += f"There are {nu_labels.max() + 1} classes:\n"
                prompt += (str(label_names) + "\n")
        prompt += f"Output the most 1 possible category of this product as a python list, like ['{instruction_format}']"
        prompt += "\nResult:"
        prompts.append(prompt)
    if mapping != None:
        return prompts, selected_y, selected_category, human_label_names
    else:
        return prompts, selected_y, selected_category





    



def transform_category(category):
    parts = category.split()
    if len(parts) != 3 or parts[0].lower() != 'arxiv' or parts[1].lower() != 'cs':
        raise ValueError("Input should be in the format 'arxiv cs xx'")
    return "{} {}.{}".format(parts[0], parts[1], parts[2].upper())


def print_to_file(lists, output_name = "abc.txt"):
    with open(output_name, "w") as f:
        for line in lists:
            f.write(line.replace('\n', ''))
            f.write("\n")




class ComprehensiveStudy:
    def __init__(self):
        # self.datasets = ['cora', 'citeseer', 'pubmed', 'arxiv', 'products']
        self.llm = Bedrock(max_tokens=4096)
        self.datasets = ['amazonratings']
        self.arxiv_mapping, self.citeseer_mapping, self.pubmed_mapping, self.cora_mapping, self.products_mapping, self.bookhis_mapping, self.amazonratings_mapping = load_mapping()
        self.split = "fixed"
        self.seeds = [0,1,2]
        self.sample_num = 100

    def prepare_dataset(self, dataset_name, split, seed):
        set_seed_config(seed)
        dataset = torch.load(f"./cache_data_minilm/{dataset_name}/processed/geometric_data_processed.pt", map_location = 'cpu')[0]
        texts = torch.load(f"./cache_data_minilm/{dataset_name}/processed/texts.pkl", map_location = 'cpu')[0]
        if dataset_name == 'amazonratings':
            dataset.label_names = ['5score', '4.5 score', '4 score', '3.5 score', '0-3 score']
        elif dataset_name == 'bookhis':
            dataset.label_names = ['World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa', 'Ancient Civilizations', 'Middle East', 'Historical Study & Educational Resources', 'Australia & Oceania', 'Arctic & Antarctica']
        dataset.raw_texts = texts    
        sampled_test_node_idxs, train_node_idxs = get_sampled_nodes(dataset, self.sample_num)

        print(f"{dataset_name} data processed!")
        instruction = 'XX'
        if dataset_name == "arxiv":
            mapping = self.arxiv_mapping
        elif dataset_name == 'citeseer':
            mapping = self.citeseer_mapping
        elif dataset_name == 'pubmed':
            mapping = self.pubmed_mapping
        elif dataset_name == 'cora':
            mapping = self.cora_mapping
        elif dataset_name == 'products':
            mapping = self.products_mapping
        elif dataset_name == 'bookhis':
            mapping = self.bookhis_mapping
        elif dataset_name == 'amazonratings':
            mapping = self.amazonratings_mapping

        return mapping, dataset, sampled_test_node_idxs, train_node_idxs

    def zero_shot(self, dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name, instruction='XX'):
        zero_shot_prompt_human_top1, y, cat_names, human_labels = prompt_zero_shot(dataset, sampled_test_node_idxs, train_node_idxs, topk=False, need_class=True, instruction_format=instruction, mapping = mapping)
        results = []
        for x in tqdm(zero_shot_prompt_human_top1):
            results.append(self.llm.generate_text(x))
        human_zero_shot_top1_pred_texts = results
        top1_acc = topk_accuracy(human_zero_shot_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human zero shot top1 acc: {top1_acc}")
    
    def few_shot(self, dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name, instruction='XX'):
        few_shot_samples = few_shot()
        few_shot_prompt_human_top1, y, cat_names, human_labels = prompt_few_shot(dataset, few_shot_samples, sampled_test_node_idxs, train_node_idxs, topk = False, need_class = True, instruction_format = instruction, mapping = mapping, cot = False, dataset_name = dataset_name, shots = 3)
        results = []
        for x in tqdm(few_shot_prompt_human_top1):
            results.append(self.llm.generate_text(x))
        human_few_shot_top1_pred_texts = results
        top1_acc = topk_accuracy(human_few_shot_top1_pred_texts, y, human_labels, topk = False)
        print(f"{dataset_name} human few shot top1 acc: {top1_acc}")
        return few_shot_samples
    
    




    def full_run(self):
        for seed in self.seeds:
            for dataset_name in self.datasets:
                mapping, dataset, sampled_test_node_idxs, train_node_idxs = self.prepare_dataset(dataset_name, self.split, seed)
                self.zero_shot(dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)
                few_shot_samples = self.few_shot(dataset, sampled_test_node_idxs, train_node_idxs, mapping, seed, dataset_name)


def main():
    study = ComprehensiveStudy()
    study.full_run()
    # different_prompt_try()


if __name__ == '__main__':
    main()