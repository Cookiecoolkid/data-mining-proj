(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 1000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 25.39 ms
RandomWalk-NSG: Recall@10 = 0.6140, Time = 14.90 ms
RandomWalk-HNSW: Recall@10 = 0.6110, Time = 10.54 ms
WL-Flat: Recall@10 = 1.0000, Time = 35.65 ms
WL-NSG: Recall@10 = 0.1870, Time = 10.83 ms
WL-HNSW: Recall@10 = 0.2660, Time = 14.87 ms
Plot saved as recall_vs_time.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 1000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 32.90 ms
RandomWalk-NSG: Recall@10 = 0.8960, Time = 19.10 ms
RandomWalk-HNSW: Recall@10 = 0.9450, Time = 9.22 ms
RandomWalk-IVFPQ: Recall@10 = 0.9740, Time = 5.47 ms
WL-Flat: Recall@10 = 1.0000, Time = 36.56 ms
WL-NSG: Recall@10 = 0.3440, Time = 13.80 ms
WL-HNSW: Recall@10 = 0.3690, Time = 11.15 ms
WL-IVFPQ: Recall@10 = 0.8060, Time = 4.42 ms
Plot saved as recall_vs_time_k10_num_subgraphs1000_subgraph_size20.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 1000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 24.08 ms
RandomWalk-NSG: Recall@10 = 0.5310, Time = 14.86 ms
RandomWalk-HNSW: Recall@10 = 0.5380, Time = 10.91 ms
RandomWalk-IVFPQ: Recall@10 = 1.0000, Time = 4.48 ms
WL-Flat: Recall@10 = 1.0000, Time = 53.65 ms
WL-NSG: Recall@10 = 0.2120, Time = 15.77 ms
WL-HNSW: Recall@10 = 0.2880, Time = 11.09 ms
WL-IVFPQ: Recall@10 = 0.9840, Time = 1.53 ms
Plot saved as recall_vs_time_k10_num_subgraphs1000_subgraph_size10.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 1000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 27.88 ms
RandomWalk-NSG: Recall@10 = 0.8910, Time = 9.91 ms
RandomWalk-HNSW: Recall@10 = 0.9240, Time = 13.83 ms
RandomWalk-IVFPQ: Recall@10 = 0.8920, Time = 5.20 ms
WL-Flat: Recall@10 = 1.0000, Time = 24.24 ms
WL-NSG: Recall@10 = 0.2140, Time = 9.89 ms
WL-HNSW: Recall@10 = 0.3160, Time = 8.89 ms
WL-IVFPQ: Recall@10 = 0.7370, Time = 5.01 ms
Plot saved as recall_vs_time_k10_num_subgraphs1000_subgraph_size50.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 2000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 51.16 ms
RandomWalk-NSG: Recall@10 = 0.4050, Time = 9.88 ms
RandomWalk-HNSW: Recall@10 = 0.4190, Time = 13.91 ms
RandomWalk-IVFPQ: Recall@10 = 0.9960, Time = 4.49 ms
WL-Flat: Recall@10 = 1.0000, Time = 33.14 ms
WL-NSG: Recall@10 = 0.4550, Time = 1.29 ms
WL-HNSW: Recall@10 = 0.5710, Time = 10.90 ms
WL-IVFPQ: Recall@10 = 0.8350, Time = 8.25 ms
Plot saved as recall_vs_time_k10_num_subgraphs2000_subgraph_size10.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 2000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 39.35 ms
RandomWalk-NSG: Recall@10 = 0.8490, Time = 10.90 ms
RandomWalk-HNSW: Recall@10 = 0.9190, Time = 10.92 ms
RandomWalk-IVFPQ: Recall@10 = 0.9400, Time = 1.15 ms
WL-Flat: Recall@10 = 1.0000, Time = 32.60 ms
WL-NSG: Recall@10 = 0.4560, Time = 17.49 ms
WL-HNSW: Recall@10 = 0.5490, Time = 10.87 ms
WL-IVFPQ: Recall@10 = 0.7020, Time = 9.08 ms
Plot saved as recall_vs_time_k10_num_subgraphs2000_subgraph_size20.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 2000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 7.83 ms
RandomWalk-NSG: Recall@10 = 0.9080, Time = 10.90 ms
RandomWalk-HNSW: Recall@10 = 0.9470, Time = 13.93 ms
RandomWalk-IVFPQ: Recall@10 = 0.9190, Time = 3.06 ms
WL-Flat: Recall@10 = 1.0000, Time = 41.91 ms
WL-NSG: Recall@10 = 0.1760, Time = 14.42 ms
WL-HNSW: Recall@10 = 0.5290, Time = 11.16 ms
WL-IVFPQ: Recall@10 = 0.6680, Time = 4.35 ms
Plot saved as recall_vs_time_k10_num_subgraphs2000_subgraph_size50.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 5000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/anaconda3/envs/dm/lib/python3.10/site-packages/grakel/kernels/kernel.py:202: RuntimeWarning: invalid value encountered in sqrt
  return km / np.sqrt(np.outer(self._X_diag, self._X_diag))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 78.46 ms
RandomWalk-NSG: Recall@10 = 0.5410, Time = 11.89 ms
RandomWalk-HNSW: Recall@10 = 0.4930, Time = 0.45 ms
RandomWalk-IVFPQ: Recall@10 = 0.7910, Time = 5.43 ms
WL-Flat: Recall@10 = 1.0000, Time = 9.23 ms
WL-NSG: Recall@10 = 0.5220, Time = 0.62 ms
WL-HNSW: Recall@10 = 0.6940, Time = 1.47 ms
WL-IVFPQ: Recall@10 = 0.6110, Time = 10.76 ms
Plot saved as recall_vs_time_k10_num_subgraphs5000_subgraph_size10.png

(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_recall_test.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampled 8000 subgraphs.
Processing Kernel: RandomWalk
/home/cookiecoolkid/course/data-mining/src/plot_ogb_arxiv_recall_test.py:96: RuntimeWarning: invalid value encountered in sqrt
  diag = np.sqrt(np.diag(K))
Testing Index: Flat with Kernel: RandomWalk
Building index of type: Flat with dimension: 128
Index built for Flat with kernel RandomWalk.
Testing Index: NSG with Kernel: RandomWalk
Building index of type: NSG with dimension: 128
Index built for NSG with kernel RandomWalk.
Testing Index: HNSW with Kernel: RandomWalk
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel RandomWalk.
Testing Index: IVFPQ with Kernel: RandomWalk
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel RandomWalk.
Processing Kernel: WL
Testing Index: Flat with Kernel: WL
Building index of type: Flat with dimension: 128
Index built for Flat with kernel WL.
Testing Index: NSG with Kernel: WL
Building index of type: NSG with dimension: 128
Index built for NSG with kernel WL.
Testing Index: HNSW with Kernel: WL
Building index of type: HNSW with dimension: 128
Index built for HNSW with kernel WL.
Testing Index: IVFPQ with Kernel: WL
Building index of type: IVFPQ with dimension: 128
Index built for IVFPQ with kernel WL.
RandomWalk-Flat: Recall@10 = 1.0000, Time = 10.30 ms
RandomWalk-NSG: Recall@10 = 0.0000, Time = 0.38 ms
RandomWalk-HNSW: Recall@10 = 0.0000, Time = 0.51 ms
RandomWalk-IVFPQ: Recall@10 = 0.9960, Time = 8.09 ms
WL-Flat: Recall@10 = 1.0000, Time = 11.56 ms
WL-NSG: Recall@10 = 0.6460, Time = 9.90 ms
WL-HNSW: Recall@10 = 0.7460, Time = 0.88 ms
WL-IVFPQ: Recall@10 = 0.5120, Time = 1.65 ms
Plot saved as recall_vs_time_k10_num_subgraphs8000_subgraph_size10.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ 



============================ Memory ============================
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_memory.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampling: 1000 subgraphs, size=10
Profiling memory for kernel embedding: RandomWalk
/home/cookiecoolkid/course/data-mining/src/plot_ogb_arxiv_memory.py:76: RuntimeWarning: invalid value encountered in sqrt
  diag = np.sqrt(np.diag(K))
Memory: 978.57 MB
/home/cookiecoolkid/course/data-mining/src/plot_ogb_arxiv_memory.py:76: RuntimeWarning: invalid value encountered in sqrt
  diag = np.sqrt(np.diag(K))
Profiling memory for index build: RandomWalk-Flat
Memory: 934.23 MB
Profiling memory for index build: RandomWalk-HNSW
Memory: 935.63 MB
Profiling memory for index build: RandomWalk-NSG
Memory: 939.54 MB
Profiling memory for index build: RandomWalk-IVFPQ
Memory: 948.13 MB
Profiling memory for kernel embedding: WL
Memory: 1011.38 MB
Profiling memory for index build: WL-Flat
Memory: 1002.30 MB
Profiling memory for index build: WL-HNSW
Memory: 1002.30 MB
Profiling memory for index build: WL-NSG
Memory: 1002.46 MB
Profiling memory for index build: WL-IVFPQ
Memory: 1004.96 MB
Plot saved: memory_num_subgraphs1000_subgraph_size10.png

cookiecoolkid@Cookie:~/course/data-mining/src$ conda activate dm
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_memory.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampling: 2000 subgraphs, size=10
Profiling memory for kernel embedding: RandomWalk
Memory: 1121.66 MB
Profiling memory for index build: RandomWalk-Flat
Memory: 969.25 MB
Profiling memory for index build: RandomWalk-HNSW
Memory: 970.50 MB
Profiling memory for index build: RandomWalk-NSG
Memory: 973.78 MB
Profiling memory for index build: RandomWalk-IVFPQ
Memory: 974.25 MB
Profiling memory for kernel embedding: WL
Memory: 1167.21 MB
Profiling memory for index build: WL-Flat
Memory: 974.34 MB
Profiling memory for index build: WL-HNSW
Memory: 974.34 MB
Profiling memory for index build: WL-NSG
Memory: 976.38 MB
Profiling memory for index build: WL-IVFPQ
Memory: 976.38 MB
Plot saved: memory_num_subgraphs2000_subgraph_size10.png
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ 
(dm) cookiecoolkid@Cookie:~/course/data-mining/src$ python plot_ogb_arxiv_memory.py 
Graph loaded. Nodes: 169343 Edges: 1157799
Sampling: 5000 subgraphs, size=10
Profiling memory for kernel embedding: RandomWalk
Memory: 2148.99 MB
Profiling memory for index build: RandomWalk-Flat
Memory: 984.74 MB
Profiling memory for index build: RandomWalk-HNSW
Memory: 985.52 MB
Profiling memory for index build: RandomWalk-NSG
Memory: 994.27 MB
Profiling memory for index build: RandomWalk-IVFPQ
Memory: 1033.34 MB
Profiling memory for kernel embedding: WL
Memory: 2491.96 MB
Profiling memory for index build: WL-Flat
Memory: 1021.43 MB
Profiling memory for index build: WL-HNSW
Memory: 1021.43 MB
Profiling memory for index build: WL-NSG
Memory: 1025.18 MB
Profiling memory for index build: WL-IVFPQ
Memory: 1025.18 MB
Plot saved: memory_num_subgraphs5000_subgraph_size10.png

| Subgraphs Size | RandomWalk Embedding | RandomWalk-Flat | RandomWalk-HNSW | RandomWalk-NSG | RandomWalk-IVFPQ | WL Embedding | WL-Flat | WL-HNSW | WL-NSG | WL-IVFPQ |
|----------------|----------------------|-----------------|-----------------|---------------|-----------------|--------------|---------|---------|--------|---------|
| 1000           | 978.57 MB            | 934.23 MB       | 935.63 MB       | 939.54 MB     | 948.13 MB       | 1011.38 MB   | 1002.30 MB | 1002.30 MB | 1002.46 MB | 1004.96 MB |
| 2000           | 1121.66 MB           | 969.25 MB       | 970.50 MB       | 973.78 MB     | 974.25 MB       | 1167.21 MB   | 974.34 MB | 974.34 MB | 976.38 MB | 976.38 MB |
| 5000           | 2148.99 MB           | 984.74 MB       | 985.52 MB       | 994.27 MB     | 1033.34 MB      | 2491.96 MB   | 1021.43 MB | 1021.43 MB | 1025.18 MB | 1025.18 MB |






Search-Time
| 参数组合 (Subgraphs=1000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat  | WL-NSG   | WL-HNSW  | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | -------- | -------- | -------- | -------- |
| Size=10               | 24.08 ms        | 14.86 ms       | 10.91 ms        | 4.48 ms          | 53.65 ms | 15.77 ms | 11.09 ms | 1.53 ms  |
| Size=20               | 32.90 ms        | 19.10 ms       | 9.22 ms         | 5.47 ms          | 36.56 ms | 13.80 ms | 11.15 ms | 4.42 ms  |
| Size=50               | 27.88 ms        | 9.91 ms        | 13.83 ms        | 5.20 ms          | 24.24 ms | 9.89 ms  | 8.89 ms  | 5.01 ms  |


Recall
| 参数组合 (Subgraphs=1000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat | WL-NSG | WL-HNSW | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | ------- | ------ | ------- | -------- |
| Size=10               | 1.0000          | 0.5310         | 0.5380          | 1.0000           | 1.0000  | 0.2120 | 0.2880  | 0.9840   |
| Size=20               | 1.0000          | 0.8960         | 0.9450          | 0.9740           | 1.0000  | 0.3440 | 0.3690  | 0.8060   |
| Size=50               | 1.0000          | 0.8910         | 0.9240          | 0.8920           | 1.0000  | 0.2140 | 0.3160  | 0.7370   |


Search-Time
| 参数组合 (Subgraphs=2000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat  | WL-NSG   | WL-HNSW  | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | -------- | -------- | -------- | -------- |
| Size=10               | 51.16 ms        | 9.88 ms        | 13.91 ms        | 4.49 ms          | 33.14 ms | 1.29 ms  | 10.90 ms | 8.25 ms  |
| Size=20               | 39.35 ms        | 10.90 ms       | 10.92 ms        | 1.15 ms          | 32.60 ms | 17.49 ms | 10.87 ms | 9.08 ms  |
| Size=50               | 7.83 ms         | 10.90 ms       | 13.93 ms        | 3.06 ms          | 41.91 ms | 14.42 ms | 11.16 ms | 4.35 ms  |

Recall
| 参数组合 (Subgraphs=2000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat | WL-NSG | WL-HNSW | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | ------- | ------ | ------- | -------- |
| Size=10               | 1.0000          | 0.4050         | 0.4190          | 0.9960           | 1.0000  | 0.4550 | 0.5710  | 0.8350   |
| Size=20               | 1.0000          | 0.8490         | 0.9190          | 0.9400           | 1.0000  | 0.4560 | 0.5490  | 0.7020   |
| Size=50               | 1.0000          | 0.9080         | 0.9470          | 0.9190           | 1.0000  | 0.1760 | 0.5290  | 0.6680   |

| 参数组合 (Subgraphs=5000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat | WL-NSG | WL-HNSW | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | ------- | ------ | ------- | -------- |
| Size=10               | 78.46 ms          | 11.89 ms         | 0.45 ms          | 5.43 ms           | 9.23 ms  | 0.62 ms | 1.47 ms  | 10.76 ms   |
| Size=20               | 4.42 ms          | 0.13 ms         | 0.15 ms          | 4.28 ms           | 6.82 ms  | 11.98 ms | 0.36 ms  | 3.82 ms   |
| Size=50               | 32.56 ms          | 9.11 ms         | 14.23 ms          | 12.02 ms           | 69.97 ms  | 10.94 ms | 0.92 ms  | 5.26 ms   |

| 参数组合 (Subgraphs=5000) | RandomWalk-Flat | RandomWalk-NSG | RandomWalk-HNSW | RandomWalk-IVFPQ | WL-Flat | WL-NSG | WL-HNSW | WL-IVFPQ |
| --------------------- | --------------- | -------------- | --------------- | ---------------- | ------- | ------ | ------- | -------- |
| Size=10               | 1.0000          | 0.5410         | 0.4930          | 0.7910           | 1.0000  | 0.5220 | 0.6940  | 0.6110   |
| Size=20               | 1.0000          | 0.0000         | 0.0000          | 0.7120           | 1.0000  | 0.6190 | 0.7810  | 0.5860   |
| Size=50               | 1.0000          | 0.0020         | 0.0030          | 0.9920           | 1.0000  | 0.6340 | 0.6260  | 0.6170   |
