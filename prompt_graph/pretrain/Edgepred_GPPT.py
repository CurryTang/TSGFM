import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from prompt_graph.data import load4link_prediction_multi_graph, load4link_prediction_single_graph
from torch.optim import Adam
import time
from .base import PreTrain
import os
from graphmae.config import DATASET
import numpy as np
from tqdm import tqdm

class Edgepred_GPPT(PreTrain):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)  
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim) 
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.output_dim).to(self.device)  

    def generate_loader_data(self):
        loaders = []
        self.datas = []
        for d in self.dataset_name:
            if DATASET[d]['level'] == 'node':
                data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(d)  
                # self.data.to(self.device) 
                self.datas.append(data)
                edge_index = edge_index.transpose(0, 1)
                data = TensorDataset(edge_label, edge_index)
                loaders.append(DataLoader(data, batch_size=4096, shuffle=True))
            
            elif DATASET[d]['level'] == 'graph':
                data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(d)          
                # self.data.to(self.device) 
                self.datas.append(data)
                edge_index = edge_index.transpose(0, 1)
                data = TensorDataset(edge_label, edge_index)
                loaders.append(DataLoader(data, batch_size=256, shuffle=True))
        return loaders
      
    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device

        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.gnn.train()
        avg_loss = []
        for i, dl in enumerate(self.dataloader):
            with tqdm(total=len(dl), desc='Pretrain') as pbar:
                for step, (batch_edge_label, batch_edge_index) in enumerate(dl):
                    self.optimizer.zero_grad()

                    data = self.datas[i]
                    batch_edge_label = batch_edge_label.to(device)
                    batch_edge_index = batch_edge_index.to(device)
                    
                    out = self.gnn(data.x.to(device), data.edge_index.to(device))
                    node_emb = self.graph_pred_linear(out)
                
                    batch_edge_index = batch_edge_index.transpose(0,1)
                    batch_pred_log = self.gnn.decode(node_emb,batch_edge_index).view(-1)
                    loss = criterion(batch_pred_log, batch_edge_label)

                    loss.backward()
                    self.optimizer.step()

                    accum_loss += float(loss.detach().cpu().item())
                    total_step += 1
                    pbar.update(1)

        return accum_loss / total_step

    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = 1000000
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()
            print(f"[Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")
            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_GPPT', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))

