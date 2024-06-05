## Split several segments of MAG240M for pre-training
import sys 
import os
os.environ['METIS_DLL'] = "/mnt/home/chenzh85/anaconda3/envs/acl24/lib/libmetis.so"
sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))
import dgl
import dgl.function as fn
import collections
import networkx as nx
import tqdm
import json
import mmap
import os.path as osp
import numpy as np

from data.mag240m import *

import random
import argparse
import shutil
from torch_geometric.utils import from_dgl


## delete data with following labels to ensure no leakage

arxiv_labels = set([0, 1, 3, 6, 9, 16, 17, 23, 24, 26,
						29, 39, 42, 47, 52, 57, 59, 63, 73, 77,
						79, 85, 86, 89, 94, 95, 105, 109, 114, 119,
						120, 122, 124, 130, 135, 137, 139, 147, 149, 152])


def get_args():
	args = argparse.ArgumentParser()
	args.add_argument("--start", type=int, default=0)
	args.add_argument("--end", type=int, default=10)
	return args.parse_args()

def keep_attrs_for_data(data):
	for k in data.keys():
		if k not in ['x', 'edge_index', 'edge_attr', 'y']:
			try:
				data[k] = None
			except Exception as e:
				pass
	return data


# def get_texts_from_very_large_files(idx, very_large_file_path):
# 	texts = ["" for _ in idx]
# 	hash_idx = {j: i for i, j in enumerate(idx)}
# 	fill = 0
# 	total_iterations = len(texts)
# 	with tqdm.tqdm(total=total_iterations) as pbar:
# 		with open(very_large_file_path, 'r') as f:
# 			mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
# 			while True:
# 				line = mm.readline()
# 				if not line:
# 					break 
# 				if fill == len(idx):
# 					break
# 				try:
# 					line = line.decode('utf-8').strip()
# 					left, right = line.split(',', maxsplit=1)
				
# 					if int(left) in hash_idx:
# 						texts[hash_idx[int(left)]] = right
# 						fill += 1
# 						pbar.update(1)
# 				except Exception as e:
# 					continue
# 			mm.close()
# 	return texts
			
				






if __name__ == '__main__':
	print("Building graph")


	params = get_args()

	# test = get_texts_from_very_large_files([0, 1, 2, 3, 4], "/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/mag240m_kddcup2021/raw/mag240m_mapping/text.csv")
	# import ipdb; ipdb.set_trace()
	dataset = MAG240MDataset(root="/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m")
	#ei_cites = dataset.edge_index("paper", "paper")
	orig_label = dataset.all_paper_label
	#orig_features = dataset.paper_feat
	total_num = len(orig_label)


	# idx2name = pd.read_csv("/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/mag240m_kddcup2021/raw/mag240m_mapping/labelidx2labelname.csv", header='infer')
	# idx2name.set_index('idx', inplace=True)
	# idx2name = idx2name.to_dict('index')
	# idx2name = {i: idx2name[i]['label_name'] for i in idx2name.keys()}
	# torch.save(idx2name, 'idx2name.pt')

	# g = dgl.graph((np.concatenate([ei_cites[0], ei_cites[1]]),np.concatenate([ei_cites[1], ei_cites[0]])))

	# # dgl.save_graphs('/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/dglgraph', g)

	# g, _ = dgl.load_graphs('/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/dglgraph')
	# g = g[0]

	print("Start partitioning")

	# paper_feat = dataset.paper_feat

	# # Iteratively process author features along the feature dimension.

	# dgl.distributed.partition_graph(
	#  	g, 'mag240m', num_parts = 1000, out_path = '/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m', num_hops=1,
	#  	return_mapping=False)

	# # torch.save(nodemap, '/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/nodemap.pt')
	# nodemap = torch.load('/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/nodemap.pt')
	# with open("/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/mag240m.json", 'r') as f:
	# 	json_file = json.load(f)
	# 	node_id_anchor = json_file['node_map']["_N"]
	partition_path = "/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m"

	# ## select 120 nodes in total
	random_numbers = [i for i in range(params.start, params.end)]

	for i in tqdm.tqdm(random_numbers):
		if osp.exists(f'/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/segments/partitioned_data_{i}.pt'):
			path = f'/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/part{i}'
			if osp.exists(path):
				shutil.rmtree(path)
			continue
		part_meta = dgl.distributed.load_partition("/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/mag240m.json", i)
		partition_graph = part_meta[0]
		total_node_ids = partition_graph.num_nodes()
		node_id = partition_graph.ndata[dgl.NID]
		# idx_shift = node_id_anchor[i][0]
		y = []
		# raw_x = []
		for j in tqdm.tqdm(range(total_node_ids)):
			orig_id = node_id[j]
			if orig_label[orig_id] not in arxiv_labels and not np.isnan(orig_label[orig_id]):
				y.append(orig_label[orig_id])
				#raw_x.append(torch.tensor(orig_features[orig_id].tolist()))
			else:
				## ignore halo nodes
				y.append(-1)
				#raw_x.append(torch.zeros(768, dtype=torch.float32))

		print("Load x and y")
		y = torch.tensor(y)
		# raw_x = torch.stack(raw_x)
		# partition_graph.ndata['x'] = raw_x
		partition_graph.ndata['y'] = y

		pyg_data = from_dgl(partition_graph)
		pyg_data = keep_attrs_for_data(pyg_data)
		pyg_data.node_id = node_id
		torch.save(pyg_data, f'/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/segments/partitioned_data_{i}.pt')
		path = f'/mnt/home/chenzh85/graphlang/PyGFM/datasets/mag240m/part{i}'
		# if osp.exists(path):
		# 	shutil.rmtree(path)
			




