import umap
import matplotlib.pyplot as plt
import numpy as np
import torch  # For tensor handling
import matplotlib.cm as cm
import seaborn as sns
import torch.nn as nn

def visualize_umap_datasets(datasets, dataset_labels, dataset_names, colormap='viridis', point_size=5, mode='feature'):
    """
    Visualizes multiple tensor datasets using UMAP with different colors.

    Args:
        datasets: A list of tensors, each representing an embedding matrix (X).
        dataset_names: A list of strings, corresponding to the names of the datasets.
        colormap: The matplotlib colormap to use (e.g., 'viridis', 'tab10', 'Set1').
        point_size: The size of the points in the scatter plot.
    """
    # UMAP Reduction
    reduced_embeddings = []
    for i, X in enumerate(datasets):
        reducer = umap.UMAP(n_components=2) 
        X_np = X.detach().cpu().numpy()  # Convert tensor to NumPy array
        X_np_label = dataset_labels[i].detach().cpu().numpy()
        X_np = np.concatenate((X_np, X_np_label), axis=0)
        embedding = reducer.fit_transform(X_np)
        class_num = X_np_label.shape[0]
        if 'feature' in mode:
            reduced_embeddings.append(embedding[:-class_num, :])
        elif 'label' in mode:
            reduced_embeddings.append(embedding[-class_num:, :])

    # Combine and Label
    # all_embeddings = np.vstack(reduced_embeddings)
    # labels = np.repeat(dataset_names, [len(X) for X in datasets])

    colors = cm.jet(np.linspace(0, 1, len(dataset_names)))


    # Visualization
    plt.figure(figsize=(10, 8))
    for i, x in enumerate(reduced_embeddings):
        plt.scatter(
            x[:, 0],
            x[:, 1],
            s=point_size,
            color=colors[i],
            label=dataset_names[i]
        )
    plt.legend()
    plt.title("UMAP Visualization of Multiple Datasets")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(f"umap_visualization_{dataset_names}_{mode}.pdf")
    plt.savefig(f"umap_visualization_{dataset_names}_{mode}.png")



class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def average_feature_similarity_heatmap(datasets, dataset_names, similarity_metric='mmd', mode='feature'):
    """
    Calculates the average pairwise feature similarity between datasets and plots a heatmap.

    Args:
        datasets: A list of tensors, each representing a dataset's feature matrix (X).
        dataset_names: A list of strings, corresponding to the names of the datasets.
        similarity_metric: The similarity metric to use ('cosine', 'euclidean', or 'dot_product').
    """

    num_datasets = len(datasets)
    similarity_matrix = np.zeros((num_datasets, num_datasets))

    mmd = MMDLoss()

    for i, dataset1 in enumerate(datasets):
        for j, dataset2 in enumerate(datasets):
            if similarity_metric == 'cosine':
                similarity = torch.nn.functional.cosine_similarity(dataset1, dataset2, dim=1)
            elif similarity_metric == 'euclidean':
                similarity = -torch.cdist(dataset1, dataset2, p=2)  # Negative Euclidean distance
            elif similarity_metric == 'dot_product':
                similarity = torch.matmul(dataset1, dataset2.t())
            elif similarity_metric == 'mmd':
                similarity = -mmd(dataset1, dataset2)
            else:
                raise ValueError("Invalid similarity metric")

            similarity_matrix[i, j] = similarity.mean().item()  # Average similarity
    
    short_names = [f"{name[:2]}-{name[-1]}" for name in dataset_names]  # Truncate names for display
    # Create Heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=short_names, yticklabels=short_names)
    plt.title(f"Average Pairwise Feature Similarity ({similarity_metric.upper()})")
    plt.xlabel("Dataset")
    plt.ylabel("Dataset")
    plt.savefig(f"heatmap_visualization_{dataset_names}_{mode}.pdf")
    plt.savefig(f"heatmap_visualization_{dataset_names}_{mode}.png")