import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.utils import Gprompt_link_loss
from prompt_graph.utils import edge_index_to_sparse_matrix, prepare_structured_data
from prompt_graph.data import load4link_prediction_single_graph,load4link_prediction_multi_graph
import time
from .base import PreTrain
import os
from graphmae.config import DATASET
from tqdm import tqdm

class Edgepred_Gprompt(PreTrain):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim) 
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.output_dim).to(self.device)  

    def generate_loader_data(self):
        loaders = []
        self.datas = []
        self.adjs = []
        for d in self.dataset_name:    
            if DATASET[d]['level'] == 'node':            
                data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(d)
                adj = edge_index_to_sparse_matrix(data.edge_index, data.x.shape[0])
                kdata = prepare_structured_data(data)
                loaders.append(DataLoader(TensorDataset(kdata), batch_size=4096, shuffle=True))
                self.datas.append(data)
                self.adjs.append(adj)
        
            elif DATASET[d]['level'] == 'graph':
                data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(d)          
                adj = edge_index_to_sparse_matrix(data.edge_index, data.x.shape[0])
                kdata = prepare_structured_data(data)
                loaders.append(DataLoader(TensorDataset(kdata), batch_size=64, shuffle=True))
                self.datas.append(data)
                self.adjs.append(adj)
        return loaders
    
    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device
        self.gnn.train()
        # import ipdb; ipdb.set_trace()
        for i, dl in enumerate(self.dataloader):
            with tqdm(total=len(dl), desc='Pretrain') as pbar:
                for step, batch in enumerate(dl):
            
                    self.optimizer.zero_grad()

                    batch = batch[0]
                    batch = batch.to(device)
                    
                    data = self.datas[i]
                    adj = self.adjs[i].to(device)
                    # import ipdb; ipdb.set_trace()
                    out = self.gnn(data.x.to(device), data.edge_index.to(device))
                                
                    all_node_emb = self.graph_pred_linear(out)

                    # TODO: GraphPrompt customized node embedding computation
                    all_node_emb = torch.sparse.mm(adj,all_node_emb)
            
                    node_emb = all_node_emb[batch[:, 0]]
                    pos_emb, neg_emb = all_node_emb[batch[:, 1]], all_node_emb[batch[:, 2]]

                    loss = Gprompt_link_loss(node_emb, pos_emb, neg_emb)

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
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_Gprompt', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
                
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_Gprompt', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))