from __future__ import print_function
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from grakel import GraphKernel
import faiss
import random
from ogb.nodeproppred import PygNodePropPredDataset
from memory_profiler import memory_usage

from torch_geometric.data.data import Data, DataEdgeAttr
from torch.serialization import add_safe_globals
add_safe_globals([Data, DataEdgeAttr])

import torch
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
import pandas as pd

# 手动设置参数
num_subgraphs = 2000
subgraph_size = 20

_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
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
    G = pyg_to_networkx(data)
    labels = data.y.numpy()
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
        labels = {i: str(i) for i in sg.nodes()}
        gk_graphs.append((adj_dict, labels))
    return gk_graphs

def kernel_to_embedding(K, dim=128):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    K_imputed = imputer.fit_transform(K)
    kpca = KernelPCA(n_components=dim, kernel='precomputed')
    return kpca.fit_transform(K_imputed)

def random_walk_embedding(grakel_graphs):
    rw_kernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=False)
    K = rw_kernel.fit_transform(grakel_graphs)
    diag = np.sqrt(np.diag(K))
    norm_matrix = np.outer(diag, diag)
    with np.errstate(divide='ignore', invalid='ignore'):
        K_normalized = np.divide(K, norm_matrix)
        K_normalized[np.isnan(K_normalized)] = 0.0
        K_normalized[np.isinf(K_normalized)] = 0.0
    return kernel_to_embedding(K_normalized, dim=128)

def wl_embedding(grakel_graphs):
    wl_kernel = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    K = wl_kernel.fit_transform(grakel_graphs)
    return kernel_to_embedding(K, dim=128)

def build_index(index_type, embeddings):
    dim = embeddings.shape[1]
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

def profile_memory(func):
    mem_usage = memory_usage(func, max_iterations=1, interval=0.1)
    return max(mem_usage)

if __name__ == "__main__":
    G, labels, split_idx = load_ogbn_arxiv()
    print("Graph loaded. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    print(f"Sampling: {num_subgraphs} subgraphs, size={subgraph_size}")
    subgraphs = sample_subgraphs(G, num_subgraphs=num_subgraphs, subgraph_size=subgraph_size)
    grakel_graphs = nx_to_grakel(subgraphs)

    kernels = {
        "RandomWalk": random_walk_embedding,
        "WL": wl_embedding
    }

    index_types = ["Flat", "HNSW", "NSG", "IVFPQ"]
    memory_results = []

    for kernel_name, embed_func in kernels.items():
        # Profile kernel embedding memory
        print(f"Profiling memory for kernel embedding: {kernel_name}")
        mem_kernel = profile_memory(lambda: embed_func(grakel_graphs))
        memory_results.append((kernel_name, "KernelEmbedding", mem_kernel))
        print(f"Memory: {mem_kernel:.2f} MB")

        # Get embeddings for indexing
        embeddings = embed_func(grakel_graphs)

        for index_type in index_types:
            print(f"Profiling memory for index build: {kernel_name}-{index_type}")
            mem_index = profile_memory(lambda: build_index(index_type, embeddings))
            memory_results.append((kernel_name, index_type, mem_index))
            print(f"Memory: {mem_index:.2f} MB")

    # 绘图
    plt.figure(figsize=(10, 6))
    for kernel_name, index_type, peak_mem in memory_results:
        plt.scatter(index_type, peak_mem, label=f"{kernel_name}-{index_type}", s=100)

    plt.title("Peak Memory Usage (MB)")
    plt.xlabel("Kernel-Index Type/Embedding")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"memory_num_subgraphs{num_subgraphs}_subgraph_size{subgraph_size}.png"
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.show()
