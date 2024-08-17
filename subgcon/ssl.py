## SUPPORTED SSL METHODS: DGI, GCC, BGRL

import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
import copy
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.utils import to_undirected, remove_self_loops, degree
from graphllm.utils import MP

def kmeans(X, num_clusters, distance='euclidean', device='cuda', max_iter=100, tol=1e-4):
    """
    Perform KMeans clustering on the input data X.

    Parameters:
    X : torch.Tensor
        Input data, shape [n_samples, n_features]
    num_clusters : int
        Number of clusters
    distance : str
        Distance metric ('euclidean' is currently supported)
    device : str
        Device to use ('cuda' or 'cpu')
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence

    Returns:
    cluster_ids_x : torch.Tensor
        Cluster assignment for each sample
    cluster_centers : torch.Tensor
        Cluster centers
    """

    if distance != 'euclidean':
        raise NotImplementedError("Currently only 'euclidean' distance is supported.")

    X = X.to(device)
    n_samples, n_features = X.shape

    # Randomly initialize cluster centers
    random_indices = torch.randperm(n_samples)[:num_clusters]
    cluster_centers = X[random_indices]

    for i in range(max_iter):
        # Compute distances and assign clusters
        distances = torch.cdist(X, cluster_centers)
        cluster_ids_x = torch.argmin(distances, dim=1)

        # Compute new cluster centers
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            cluster_k = X[cluster_ids_x == k]
            if len(cluster_k) > 0:
                new_cluster_centers[k] = cluster_k.mean(dim=0)

        # Check for convergence
        if torch.norm(new_cluster_centers - cluster_centers) < tol:
            break

        cluster_centers = new_cluster_centers

    return cluster_ids_x, cluster_centers


class DGIEncoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(DGIEncoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index, batch=None):
        z = self.encoder(x, edge_index)

        if batch is None:
            g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        else:
            g = self.project(torch.sigmoid(global_mean_pool(z, batch)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn

def dgi_train_step(encoder_model, contrast_model, data, optimizer):
    # model.train()
    # optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn, batch=data.batch)
    return loss


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batchnorm'):
        super().__init__()
        if norm is None:
            self.norm = lambda x: x
        elif dim is None or norm.lower() == 'none':
            self.norm = lambda x: x
        elif norm.lower() == 'batchnorm':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm.lower() == 'layernorm':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class BGRLEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(BGRLEncoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.batch_norm = Normalize(hidden_dim, norm=predictor_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        h1 = self.online_encoder(x1, edge_index1, edge_weight1)
        h1 = self.batch_norm(h1)
        h1_online = self.projection_head(h1)
        h2 = self.online_encoder(x2, edge_index2, edge_weight2)
        h2 = self.batch_norm(h2)
        h2_online = self.projection_head(h2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            h1 = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            h1 = self.batch_norm(h1)
            h1_target = self.projection_head(h1)
            h2 = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            h2 = self.batch_norm(h2)
            h2_target = self.projection_head(h2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target

def bgrl_train_step(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    # loss.backward()
    # optimizer.step()
    # encoder_model.update_target_encoder(0.99)
    return loss 


class EdgePred_GPPT(torch.nn.Module):
    def __init__(self, gnn_encoder, input_dim, hidden_dim, output_dim):
        super(EdgePred_GPPT, self).__init__()
        
        self.gnn_encoder = gnn_encoder
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_pred_linear = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, x, edge_index, edge_label, edge_weight=None):
        out = self.gnn_encoder(x, edge_index, edge_weight)
        node_emb = self.graph_pred_linear(out)
        # edge_index = edge_index.transpose(0, 1)
        pred_log = (node_emb[edge_index[0]] * node_emb[edge_index[1]]).sum(dim=1).view(-1)
        # import ipdb; ipdb.set_trace()
        loss = self.criterion(pred_log, edge_label.float())
        return loss

def pretrain_gppt(model, data):
    model.train()
    return model(data.x, data.edge_index, data.edge_label, data.edge_weight)

def pretrain_gprompt(model, data):
    model.train()
    return model(data.x, data.edge_index, data.edge_weight)


class SimpleMeanConv(MessagePassing):
    def __init__(self):
        super(SimpleMeanConv, self).__init__(aggr='mean') 
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    def message(self, x_j):
        return x_j




class GPPT(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes, device):
        super(GPPT, self).__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.structure_token = torch.nn.Linear(n_hidden, center_num, bias = False).to(device)
        self.task_token = torch.nn.ModuleList()
        for _ in range(center_num):
            self.task_token.append(torch.nn.Linear(n_hidden, n_classes, bias = False))
        self.task_token = self.task_token.to(device)
    
    @torch.no_grad()
    def weight_init(self, h, edge_index, label, index):
        conv = SimpleMeanConv()
        h = conv(h, edge_index)
        features = h[index]
        labels = label[index.long()]
        _, cluster = kmeans(features, self.center_num)
        self.structure_token.weight.data = torch.tensor(cluster)
        p = []
        for i in range(self.n_classes):
            p.append(features[labels == i].mean(dim=0).view(1, -1))
        temp = torch.cat(p, dim=0)
        for i in range(self.center_num):
            self.task_token[i].weight.data = temp

    def update_structuretoken_weight(self, h):
        _, cluster = kmeans(h, self.center_num)
        self.structure_token.weight.data = torch.tensor(cluster) 
    
    def get_task_token(self):
        parmas = []
        for name, param in self.named_parameters():
            if 'task_token' in name:
                parmas.append(param)
        return parmas

    def get_structure_token(self):
        params = []
        for name, param in self.named_parameters():
            if 'structure_token' in name:
                params.append(param)
        return params

    def get_mid_h(self):
        return self.fea
    
    def forward(self, h, edge_index):
        conv = SimpleMeanConv()
        h = conv(h, edge_index)
        self.fea = h
        out = self.structure_token(h)
        index = torch.argmax(out, dim = 1)
        out = torch.zeros(h.shape[0], self.n_classes).to(self.device)   
        for i in range(self.center_num):
            out[index == i] = self.task_token[i](h[index == i])
        return out

def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))

def gppt_train_step(model, prompt, data, pg_optimizer, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    prompt.train()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    # import ipdb; ipdb.set_trace()
    data.is_train = data.is_train.to(device)
    data.y = data.y.to(device)
    h = model(data.x, data.edge_index, data.edge_attr)
    out = prompt(h, data.edge_index)
    loss = criterion(out[data.is_train], data.y[data.is_train])
    loss = loss + 0.001 * constraint(device, prompt.get_task_token())
    pg_optimizer.zero_grad()
    loss.backward()
    pg_optimizer.step()
    mid_h = prompt.get_mid_h()
    prompt.update_structuretoken_weight(mid_h)
    return loss.item()

def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts

def gprompt_train_step(model, prompt, data, pg_optimizer, device, level='node'):
    pg_optimizer.zero_grad() 
    batch = data.to(device)
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    node_emb = prompt(out)
    graph_emb = global_mean_pool(node_emb, batch.batch)
    if level == 'node':
        out = node_emb
    else:
        out = graph_emb
    # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
    center, class_counts = center_embedding(out, batch.y, prompt.output_dim)
    delta_accumulated_centers = center * class_counts
    delta_accumulated_counts = class_counts
    loss = gprompt_tuning_loss(out, center, batch.y, 0.1) 
    loss.backward()  
    pg_optimizer.step()  
    return loss.item(), delta_accumulated_centers, delta_accumulated_counts


class GPrompt(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GPrompt, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim)) 
        self.max_n_num = input_dim
        self.output_dim = output_dim
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, edge_index, edge_weight=None):
        node_embeddings = node_embeddings * self.weight
        return node_embeddings

def compute_message_passing(edge_index, x, hop=2):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mp = MP()
    for _ in range(hop):
        x = mp.partition_propagate(edge_index, x=x, norm=norm, chunk_size=500, cuda=True)
    return x

class EdgePred_GPrompt(torch.nn.Module):
    def __init__(self, gnn_encoder, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_pred_linear = torch.nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        out = self.gnn_encoder(x, edge_index, edge_weight)
        node_emb = self.graph_pred_linear(out)
        all_node_emb = compute_message_passing(edge_index, node_emb)
        node_emb = all_node_emb[batch[:, 0]]
        pos_emb, neg_emb = all_node_emb[batch[:, 1]], all_node_emb[batch[:, 2]]
        loss = gprompt_link_loss(node_emb, pos_emb, neg_emb)
        return loss



def gprompt_tuning_loss(embedding, center_embedding, labels, tau):
    similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1) / tau
    exp_similarities = torch.exp(similarity_matrix)
    # Sum exponentiated similarities for the denominator
    pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
    # select the exponentiated similarities for the correct classes for the every pair (xi,yi)
    pos = exp_similarities.gather(1, labels.view(-1, 1))
    L_prompt = -torch.log(pos / pos_neg)
    loss = torch.sum(L_prompt)
    return loss

def gprompt_link_loss(node_emb, pos_emb, neg_emb, temperature=.2):
    r"""Refer to GraphPrompt original codes"""
    x = torch.exp(F.cosine_similarity(node_emb, pos_emb, dim=-1) / temperature)
    y = torch.exp(F.cosine_similarity(node_emb, neg_emb, dim=-1) / temperature)

    loss = -1 * torch.log(x / (x + y) )
    return loss.mean()


def prepare_structured_data(graph_data):
    r"""Prepare structured <i,k,j> format link prediction data"""
    node_idx = torch.LongTensor([i for i in range(graph_data.num_nodes)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([graph_data.edge_index, self_loop], dim=1)
    v, a, b = structured_negative_sampling(edge_index, graph_data.num_nodes)
    data = torch.stack([v, a, b], dim=1)
    # (num_edge, 3)
    #   for each entry (i,j,k) in data, (i,j) is a positive sample while (i,k) forms a negative sample
    return data