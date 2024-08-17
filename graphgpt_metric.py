import json
import os.path as osp
import os
import torch as th
import re
import pandas as pd
from tqdm import tqdm 
from sklearn.metrics import classification_report
import editdistance


data_list = []
datan = "amazonratings"
folder = f'/egr/research-dselab/chenzh85/nips/MyOFA/GLBench_{datan}_nc_output'
for filename in os.listdir(folder):
    if filename.endswith('.json'): 
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            data_list.extend(data)

print(data_list[1])

graph_data = th.load(f"instruct_ds/{datan}.pt")
labels = graph_data.y
class_map = {i: lb for i, lb in enumerate(graph_data.label_name)}
inverse_class_map = {}
for lb, lb_id in class_map.items():
    inverse_class_map[lb_id] = lb

def find_most_frequent_substring(long_string, short_strings):
    """
    Finds the index of the most frequent short string in the long string.

    Args:
        long_string: The long string to search.
        short_strings: A list of short strings to search for.

    Returns:
        The index of the most frequent short string in the long string, 
        or -1 if none of the short strings are present.
    """
    # Create a dictionary to store counts of each short string
    substring_counts = {substring: 0 for substring in short_strings}

    # Iterate through all possible starting positions of the short string in the long string
    for i in range(len(long_string) - max(len(substring) for substring in short_strings) + 1):
        # Check if any short string matches the current substring in the long string
        for substring, count in substring_counts.items():
            if long_string[i:i+len(substring)] == substring:
                substring_counts[substring] += 1

    # Find the short string with the highest count
    most_frequent_substring = max(substring_counts, key=substring_counts.get, default=None)

    # Return the index of the most frequent substring or -1 if none is found
    if most_frequent_substring:
        return short_strings.index(most_frequent_substring)
    else:
        return -1

def find_labels(long_string, short_strings):
    x = [editdistance.distance(long_string, short_string) for short_string in short_strings]
    return x.index(min(x))


topk = 1

correct = 0
total = len(data_list)

trues = []
preds = []

for instruct_item in tqdm(data_list): 
    nid = instruct_item['node_idx']
    gpt_res = instruct_item['res']
    true_y = labels[nid]
    pred_y = -1
    pred_y = find_most_frequent_substring(gpt_res, graph_data.label_name)
    # pred_y = find_labels(gpt_res, graph_data.label_name)
    #import ipdb; ipdb.set_trace()

    trues.append(true_y.item())
    correct = correct + 1 if true_y == pred_y else correct
acc = correct / total
print("Accuracy:", acc)