# nsg_wl_faiss.py
from __future__ import print_function
import networkx as nx
import numpy as np
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

def build_nsg_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexNSGFlat(dim, 32)
    index.train(embeddings.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    return index

def compute_recall(pred_indices, gt_indices):
    recalls = []
    for pred, gt in zip(pred_indices, gt_indices):
        correct = len(set(pred) & set(gt))
        recalls.append(correct / float(len(gt)))
    return np.mean(recalls)

if __name__ == "__main__":
    graph_path = '/home/cookiecoolkid/datasets/LiveJournal/com-lj.ungraph.txt'
    G = load_graph(graph_path)

    print("Graph loaded. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    subgraphs = sample_subgraphs(G)
    print("Sampled", len(subgraphs), "subgraphs.")

    grakel_graphs = nx_to_grakel(subgraphs)
    embeddings = wl_embedding(grakel_graphs)

    print("Embeddings shape:", embeddings.shape)

    query_embeddings = embeddings[:100]
    k = 10

    gt_indices = build_groundtruth(embeddings, query_embeddings, k)
    nsg_index = build_nsg_index(embeddings)

    pred_distances, pred_indices = nsg_index.search(query_embeddings.astype(np.float32), k)

    recall = compute_recall(pred_indices, gt_indices)

    print("NSG + WL Recall@{}: {:.4f}".format(k, recall))
