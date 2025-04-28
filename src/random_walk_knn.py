# random_walk_faiss.py
from __future__ import print_function
import networkx as nx
import numpy as np
from grakel import GraphKernel
import faiss
import random

# 1. 读取 LiveJournal 图
def load_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G

# 2. 采样子图（subgraph sampling）
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

# 3. 把 NetworkX 子图转成 Grakel 格式
def nx_to_grakel(subgraphs):
    gk_graphs = []
    for sg in subgraphs:
        g = {i: list(sg.neighbors(i)) for i in sg.nodes()}
        gk_graphs.append(g)
    return gk_graphs

# 4. Random Walk Kernel Embedding
def random_walk_embedding(grakel_graphs):
    rw_kernel = GraphKernel(kernel={"name": "random_walk"}, normalize=True)
    K = rw_kernel.fit_transform(grakel_graphs)
    return K

# 5. 使用 Faiss 建立KNN索引
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index

# 6. 检索测试
def knn_search(index, query_vectors, k=10):
    distances, indices = index.search(query_vectors.astype(np.float32), k)
    return distances, indices

if __name__ == "__main__":
    graph_path = '/home/cookiecoolkid/datasets/LiveJournal/com-lj.ungraph.txt'
    G = load_graph(graph_path)

    print("Graph loaded. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    subgraphs = sample_subgraphs(G)
    print("Sampled", len(subgraphs), "subgraphs.")

    grakel_graphs = nx_to_grakel(subgraphs)
    embeddings = random_walk_embedding(grakel_graphs)

    print("Embeddings shape:", embeddings.shape)

    index = build_faiss_index(embeddings)

    # 用部分子图做query测试
    query_embeddings = embeddings[:10]
    distances, indices = knn_search(index, query_embeddings)

    print("KNN Search Result:")
    for i, (dists, inds) in enumerate(zip(distances, indices)):
        print("Query", i, "neighbors:", inds, "distances:", dists)
