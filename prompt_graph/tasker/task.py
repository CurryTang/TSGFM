import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.prompt import GPF, GPF_plus, HeavyPrompt, Gprompt, GPPTPrompt
from prompt_graph.prompt import featureprompt, downprompt
from torch import nn, optim
from prompt_graph.utils import Gprompt_tuning_loss
import numpy as np

class BaseTask:
    def __init__(self, pre_train_model_path='None', gnn_type='TransformerConv',
                 hid_dim = 128, num_layer = 2, dataset_name='Cora', prompt_type='None', epochs=100, shot_num=10, device : int = 5, lr =0.001, wd = 5e-4,
                 batch_size = 16, search = False):
        
        self.pre_train_model_path = pre_train_model_path
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        self.device = torch.device('cuda:'+ str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.search = search
        self.initialize_lossfn()

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.prompt_type == 'Gprompt':
            self.criterion = Gprompt_tuning_loss()

    def initialize_optimizer(self):
        if self.prompt_type == 'None':
            if self.pre_train_model_path == 'None':
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            else:
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                # self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)

        elif self.prompt_type == 'All-in-one':
            self.pg_opi = optim.Adam( self.prompt.parameters(), lr=1e-6, weight_decay= self.wd)
            self.answer_opi = optim.Adam( self.answering.parameters(), lr=self.lr, weight_decay= self.wd)
        elif self.prompt_type in ['GPF', 'GPF-plus']:
            model_param_group = []
            model_param_group.append({"params": self.prompt.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['Gprompt']:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.prompt_type in ['GPPT']:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=2e-3, weight_decay=5e-4)
        elif self.prompt_type == 'MultiGprompt':
            self.optimizer = optim.Adam([*self.DownPrompt.parameters(),*self.feature_prompt.parameters()], lr=self.lr)

    def initialize_prompt(self):
        if self.prompt_type == 'None':
            self.prompt = None
        elif self.prompt_type == 'GPPT':
            if(self.task_type=='NodeTask'):
                if self.dataset_name == 'Texas':
                    self.prompt = GPPTPrompt(self.hid_dim, 5, self.output_dim, device = self.device)
                else:
                    self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device = self.device)
            elif(self.task_type=='GraphTask'):
                self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device = self.device)                
        elif self.prompt_type =='All-in-one':
            # lr, wd = 0.001, 0.00001
            # self.prompt = LightPrompt(token_dim=self.input_dim, token_num_per_group=100, group_num=self.output_dim, inner_prune=0.01).to(self.device)
            if(self.task_type=='NodeTask'):
                self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)
            elif(self.task_type=='GraphTask'):
                self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)
        elif self.prompt_type == 'GPF':
            self.prompt = GPF(self.input_dim).to(self.device)
        elif self.prompt_type == 'GPF-plus':
            self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
        # elif self.prompt_type == 'sagpool':
        #     self.prompt = SAGPoolPrompt(self.input_dim , num_clusters=5, ratio=0.5).to(self.device)
        # elif self.prompt_type == 'diffpool':
        #     self.prompt = DiffPoolPrompt(self.input_dim, num_clusters=5 ).to(self.device)
        elif self.prompt_type == 'Gprompt':
            self.prompt = Gprompt(self.hid_dim).to(self.device)
        else:
            raise KeyError(" We don't support this kind of prompt.")

    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)
        print(self.gnn)

        if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
            if self.gnn_type not in self.pre_train_model_path :
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            # if self.dataset_name not in self.pre_train_model_path :
            #     raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location='cpu'))
            self.gnn.to(self.device)       
            print("Successfully loaded pre-trained weights!")

    def return_pre_train_type(self, pre_train_model_path):
        names = ['None', 'DGI', 'GraphMAE','Edgepred_GPPT', 'Edgepred_Gprompt','GraphCL', 'SimGRACE']
        for name in names:
            if name  in  pre_train_model_path:
                return name


      
 
            
      