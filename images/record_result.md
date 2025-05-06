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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 1000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 1000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
WARNING clustering 2000 points to 100 centroids: please provide at least 3900 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
WARNING clustering 2000 points to 256 centroids: please provide at least 9984 training points
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
