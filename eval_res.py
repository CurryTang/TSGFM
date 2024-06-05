import random
import os.path as osp
import torch
import json
import argparse
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from difflib import SequenceMatcher
import pandas as pd

def similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def find_closest_label(text, labels):
  """Finds the element in labels with the smallest edit distance to text.

  Args:
    text: The string to compare to labels.
    labels: A list of strings representing labels.

  Returns:
    The element in labels with the smallest edit distance to text.
  """

  closest_label = None
  max_sim = -1 

  for label in labels:
    distance = similarity(text, label)
    if distance > max_sim:
      max_sim = distance
      closest_label = label

  return closest_label


def set_label_names(data, label_csv_path):
    label_pd = pd.read_csv(label_csv_path)
    if hasattr(data, 'label_names'):
        return data 
    label_names = label_pd['name'].tolist()
    data.label_names = label_names
    return data



def eval_nc(res_path, data_path):
    data=torch.load(data_path)[0]
    data_dir_path = osp.dirname(data_path)
    set_label_names(data, osp.join(data_dir_path, 'categories.csv'))
    labels=data.label_names
    labels = [x.lower().strip() for x in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            if isinstance(y, list):
                y=y[0]
            label=labels[y]
            match = False
            if ans.lower().strip() in labels:
                match = True
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct += 1
            if not match:
                new_ans = find_closest_label(ans, labels)
                if new_ans.lower().strip() == label.lower().strip():
                    overall_correct += 1            
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_lp(res_path):
    all_sample=0
    correct = 0
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            ans = res["text"].strip()
            label=res["gt"].strip()
            all_sample += 1
            if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                correct += 1
            if args.sample > 0 and all_sample >=  args.sample:
                break
    acc = correct / all_sample
    print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

def eval_lprank(res_path):
    all_sample=0
    correct = 0
    y_true = []
    y_pred=[]
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            logit = res["logit"]
            score = torch.softmax(torch.tensor(logit[:2]), dim=-1)[0].item()
            # score = logit[0]
            label=res["gt"].strip()
            if label == "yes":
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(score)
    auc = roc_auc_score(y_true, y_pred)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    acc = ((y_pred>0.5)==y_true).sum()/y_pred.shape[0]

    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    y_pos=y_pred[y_true==1]
    y_neg=y_pred[y_true==0]
    y_neg_sort, _ = torch.sort(y_neg)
    for n in [10,50,100,200,500,1000]:
        if n > y_neg_sort.shape[0]:
            break
        th = y_neg_sort[-n]
        h = (y_pos>th).sum()/y_pos.shape[0]
        print(f"Hits@{n}: {h:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/llaga-opt-2.7b-v1-simteg_all_origin_tape_multihop-laplacian_-1-2-10-linear-only-train-pretrain_acc1_nc_test_nc.jsonl")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--data_saved_path", type=str, default="cache_data_minilm")

    args = parser.parse_args()
    data_path = osp.join(args.data_saved_path, args.dataset, 'processed', "geometric_data_processed.pt")

    if args.task == 'nc':
        eval_nc(args.res_path, data_path)
    elif args.task == 'lp':
        eval_lprank(args.res_path) 
    else:
        raise NotImplementedError("Only support lp and nc now!")