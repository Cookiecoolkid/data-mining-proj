from __future__ import print_function
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from grakel import GraphKernel
import faiss
import random
from ogb.nodeproppred import PygNodePropPredDataset

# 🔽 注册全局允许反序列化的 PyG 类型
from torch_geometric.data.data import Data, DataEdgeAttr
from torch.serialization import add_safe_globals
add_safe_globals([Data, DataEdgeAttr])  # 必须在 PygNodePropPredDataset 初始化前执行

from torch_geometric.datasets import GEDDataset
import torch
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer

# 保存原始 torch.load 函数
_original_torch_load = torch.load

num_subgraphs = 5000
subgraph_size = 10

# 覆盖 torch.load，默认强制 weights_only=False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 应用 monkey patch
torch.load = patched_torch_load

def pyg_to_networkx(data):
    G = nx.Graph()
    edges = data.edge_index.numpy()
    for u, v in zip(edges[0], edges[1]):
        G.add_edge(u, v)
    return G

def load_ogbn_arxiv():
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    split_idx = dataset.get_idx_split()

    data, slices = torch.load(dataset.processed_paths[0])
    
    # 假设数据集中只有一个图
    G = pyg_to_networkx(data)
    labels = data.y.numpy()  # 获取标签
    
    return G, labels, split_idx

def sample_subgraphs(G, num_subgraphs=5000, subgraph_size=10):
    nodes = list(G.nodes())
    subgraphs = []
    for _ in range(num_subgraphs):
        center = random.choice(nodes)
        neighbors = nx.single_source_shortest_path_length(G, center, cutoff=2).keys()
        neighbors = list(neighbors)
        if len(neighbors) > subgraph_size:
            neighbors = random.sample(neighbors, subgraph_size)
        SG = G.subgraph(neighbors).copy()
        subgraphs.append(SG)
    return subgraphs

def nx_to_grakel(subgraphs):
    gk_graphs = []
    for sg in subgraphs:
        adj_dict = {i: list(sg.neighbors(i)) for i in sg.nodes()}
        labels = {i: str(i) for i in sg.nodes()}  # 使用节点 ID 的字符串作为标签
        gk_graphs.append((adj_dict, labels))  # 添加标签字典
    return gk_graphs

def kernel_to_embedding(K, dim=128):
    # 使用 SimpleImputer 填充 NaN 值
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    K_imputed = imputer.fit_transform(K)
    
    # 使用 KernelPCA 进行降维
    kpca = KernelPCA(n_components=dim, kernel='precomputed')
    embedding = kpca.fit_transform(K_imputed)
    return embedding

def random_walk_embedding(grakel_graphs):
    rw_kernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
    K = rw_kernel.fit_transform(grakel_graphs)
    embeddings = kernel_to_embedding(K, dim=128)  # 使用 KernelPCA 降维
    return embeddings

def wl_embedding(grakel_graphs):
    wl_kernel = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    K = wl_kernel.fit_transform(grakel_graphs)
    embeddings = kernel_to_embedding(K, dim=128)  # 使用 KernelPCA 降维
    return embeddings

# def random_walk_embedding(grakel_graphs):
#     rw_kernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
#     K = rw_kernel.fit_transform(grakel_graphs)
#     return K

# def wl_embedding(grakel_graphs):
#     wl_kernel = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
#     K = wl_kernel.fit_transform(grakel_graphs)
#     return K

def build_groundtruth(embeddings, query_vectors, k):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    _, gt_indices = index.search(query_vectors.astype(np.float32), k)
    return gt_indices

def build_index(index_type, embeddings):
    dim = embeddings.shape[1]
    print("Building index of type:", index_type, "with dimension:", dim)
    if index_type == 'Flat':
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(embeddings.astype(np.float32))
    elif index_type == 'NSG':
        index = faiss.IndexNSGFlat(dim, 32)
        index.nsg.search_L = 64
        index.add(embeddings.astype(np.float32))
    elif index_type == 'IVFPQ':
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, 100, 8, 8)
        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        index.nprobe = 10
    else:
        raise ValueError("Unknown index type:", index_type)
    return index

def compute_recall(pred_indices, gt_indices):
    recalls = []
    for pred, gt in zip(pred_indices, gt_indices):
        correct = len(set(pred) & set(gt))
        recalls.append(correct / float(len(gt)))
    return np.mean(recalls)

def test(index_type, kernel_type, embeddings, query_embeddings, gt_indices, k):
    index = build_index(index_type, embeddings)
    print(f"Index built for {index_type} with kernel {kernel_type}.")
    start = time.time()
    _, pred_indices = index.search(query_embeddings.astype(np.float32), k)
    end = time.time()
    search_time = (end - start) * 1000
    recall = compute_recall(pred_indices, gt_indices)
    return recall, search_time

if __name__ == "__main__":
    # 加载 ogbn-arxiv 数据集
    G, labels, split_idx = load_ogbn_arxiv()
    print("Graph loaded. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    # 采样子图

    subgraphs = sample_subgraphs(G, num_subgraphs, subgraph_size)
    print("Sampled", len(subgraphs), "subgraphs.")

    # 将子图转换为 Grakel 格式
    grakel_graphs = nx_to_grakel(subgraphs)

    kernels = {
        "RandomWalk": random_walk_embedding,
        "WL": wl_embedding
    }

    indices = ["Flat", "NSG", "HNSW", "IVFPQ"]
    k = 10
    results = {}

    for kernel_name, embed_func in kernels.items():
        print(f"Processing Kernel: {kernel_name}")
        embeddings = embed_func(grakel_graphs)
        query_embeddings = embeddings[:100]
        gt_indices = build_groundtruth(embeddings, query_embeddings, k)

        for index_type in indices:
            print(f"Testing Index: {index_type} with Kernel: {kernel_name}")
            recall, search_time = test(index_type, kernel_name, embeddings, query_embeddings, gt_indices, k)
            results[(kernel_name, index_type)] = (recall, search_time)

    # 画图
    plt.figure(figsize=(10, 6))

    markers = {'Flat': 'o', 'NSG': 's', 'HNSW': '^', 'IVFPQ': 'x'}
    colors = {'RandomWalk': 'blue', 'WL': 'green'}

    for (kernel_name, index_type), (recall, search_time) in results.items():
        plt.scatter(search_time, recall, marker=markers[index_type], color=colors[kernel_name],
                    label=f"{kernel_name}-{index_type}")
        print(f"{kernel_name}-{index_type}: Recall@{k} = {recall:.4f}, Time = {search_time:.2f} ms")

    plt.xlabel('Search Time (ms)')
    plt.ylabel(f'Recall@{k}')
    plt.title('Recall vs Search Time (Different Kernels and Indexes)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 构造包含参数的文件名
    filename = f"recall_vs_time_k{k}_num_subgraphs{num_subgraphs}_subgraph_size{subgraph_size}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()