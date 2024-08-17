from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
from prompt_graph.data import load4node,load4graph, split_induced_graphs
import pickle
import random
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*OpenBLAS.*") 


def load_induced_graph(dataset_name, data, device):
    # import ipdb; ipdb.set_trace()
    if isinstance(dataset_name, list):
        dataset_name = dataset_name[0]
    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min10_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=10, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
seed_everything(args.seed)


print('dataset_name', args.dataset_name)


if args.task == 'NodeTask':
    data, input_dim, output_dim = load4node(args.dataset_name[0])   
    data = data.to(args.device)
    if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
        graphs_list = load_induced_graph(args.dataset_name, data, args.device) 
    else:
        graphs_list = None 
         

if args.task == 'GraphTask':
    input_dim, output_dim, dataset = load4graph(args.dataset_name[0])

if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer,
                    gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type,
                    epochs = args.epochs, shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, data = data, input_dim = input_dim, output_dim = output_dim, graphs_list = graphs_list, 
                    normal_split = args.normal_split)


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, hid_dim = args.hid_dim, prompt_type = args.prompt_type, epochs = args.epochs,
                    shot_num = args.shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, dataset = dataset, input_dim = input_dim, output_dim = output_dim)
pre_train_type = tasker.pre_train_type


_, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _= tasker.run()
  
print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc)) 
print("Final F1 {:.4f}±{:.4f}(std)".format(f1,std_f1)) 
print("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc)) 




