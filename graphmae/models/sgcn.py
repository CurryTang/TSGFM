import torch 
from torch_geometric.nn import MessagePassing
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
import torch.nn.functional as F
from torch_geometric.utils import degree

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, mode = 'nofeat'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.mode = mode
        if mode != 'nofeat':
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr = None):
        if self.mode != 'nofeat':
            assert edge_attr is not None
            edge_embedding = self.bond_encoder(edge_attr)
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        else:
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, mode = 'nofeat'):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.mode = 'mode'
        if mode != 'nofeat':
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)


    def forward(self, x, edge_index, edge_attr = None):
        x = self.linear(x)
        if self.mode != 'nofeat':
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = None

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]


        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)
        

    def message(self, x_j, edge_attr = None, norm = None):
        if edge_attr is not None:
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, input_dim, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', mode='nofeat'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        if input_dim != -1:
            self.fc = torch.nn.Linear(input_dim, emb_dim)
            self.fce = torch.nn.Linear(input_dim, emb_dim)
        else:
            self.fc = torch.nn.Identity()
            self.fce = torch.nn.Identity()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, mode=mode))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, mode=mode))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr): 
        x = self.fc(x)
        if edge_attr is not None:
            edge_attr = self.fce(edge_attr)
        h_list = [x]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GraphEncoder(torch.nn.Module):
    ## take the idea from gcc
    def __init__(self, positional_embedding_size=32,
        llm_embedding_size=384,
        max_degree=128,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        norm=False,
        gnn_model="gcn", mode='nofeat', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ## three modes
        ## 1. atomencoder + bond encoder
        ## 2. pure structure modeling
        ## 3. llm embedding

        self.mode = mode 
        if self.mode == 'nofeat':
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        elif self.mode == 'llm':
            node_input_dim = llm_embedding_size
        elif self.mode == 'atomencoder':
            self.atom_encoder = AtomEncoder(emb_dim = node_hidden_dim)
            self.bond_encoder = BondEncoder(emb_dim = edge_hidden_dim)
            node_input_dim = -1
        self.max_degree = max_degree
        self.model = GNN_node(num_layer=num_layers, input_dim=node_input_dim, emb_dim=node_hidden_dim, gnn_type=gnn_model, mode=self.mode)
        if self.mode == 'nofeat':
            self.degree_embedding = torch.nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )
        
        self.lin_readout = torch.nn.Sequential(
            torch.nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_hidden_dim, output_dim),
        )
        self.norm = norm
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.xe
        if self.mode == 'nofeat':
            ## x is positional encoding in this case
            deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
            deg = torch.clamp(deg, 0, self.max_degree)
            deg_emb = self.degree_embedding(deg)
            x = torch.cat([x, deg_emb], dim=-1)
        elif self.mode == 'atomencoder':
            x = self.atom_encoder(x)
            edge_attr = self.bond_encoder(edge_attr)
        
        h = self.model(x, edge_index, edge_attr)
        h = self.lin_readout(h)

        return h
                        