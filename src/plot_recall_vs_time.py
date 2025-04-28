# plot_recall_vs_time.py
from __future__ import print_function
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from grakel import GraphKernel
import faiss
import random

def load_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G

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
        g = {i: list(sg.neighbors(i)) for i in sg.nodes()}
        gk_graphs.append(g)
    return gk_graphs

def random_walk_embedding(grakel_graphs):
    rw_kernel = GraphKernel(kernel={"name": "random_walk", "with_labels": False}, normalize=True)
    K = rw_kernel.fit_transform(grakel_graphs)
    return K

def wl_embedding(grakel_graphs):
    wl_kernel = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    K = wl_kernel.fit_transform(grakel_graphs)
    return K

def build_groundtruth(embeddings, query_vectors, k):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    _, gt_indices = index.search(query_vectors.astype(np.float32), k)
    return gt_indices

def build_index(index_type, embeddings):
    dim = embeddings.shape[1]
    if index_type == 'Flat':
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
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
    start = time.time()
    _, pred_indices = index.search(query_embeddings.astype(np.float32), k)
    end = time.time()
    search_time = (end - start) * 1000  # 毫秒
    recall = compute_recall(pred_indices, gt_indices)
    return recall, search_time

if __name__ == "__main__":
    graph_path = '/home/cookiecoolkid/datasets/LiveJournal/com-lj.ungraph.txt'
    G = load_graph(graph_path)
    print("Graph loaded. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    subgraphs = sample_subgraphs(G)
    print("Sampled", len(subgraphs), "subgraphs.")

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

    plt.xlabel('Search Time (ms)')
    plt.ylabel('Recall@10')
    plt.title('Recall vs Search Time (Different Kernels and Indexes)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
