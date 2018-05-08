# SCC

A serial implementation of the **Hong Algorithm** that can be used to find Strongly Connected Components (SCCs) in directed graphs. The algorithm exploits special properties of real world graphs like the power law distribution of SCCs (the number of SCCs of a particular size is inversely proportional to the size) to speed up the process of finding SCCs in large real world graphs.

#### Input
The program takes the following as input
1. Number of vertices in the graph (V)
2. Number of edges in the graph (E)
3. Text file that represents the input graph. Each line in the file represents one edge. Each line contains the from vertex of the edge and a space followed by the to vertex of the edge. The edges must be in sorted order of the 1st vertex(from vertex). The python script Sort.py can be used for this purpose.
4. Text file that represents the inverse of the input graph. This can be generated from the input graph text file and the python script reverse.py

#### Data Sets
Data sets for large real world graphs can be obtained for free from [SNAP](http://snap.stanford.edu/index.html).

#### Research Papers
[1] [On Fast Parallel Detection of Strongly Connected Components (SCC) in Small-World Graphs](https://ppl.stanford.edu/papers/sc13-hong.pdf) 

[2] [GPU Centric Extensions for Parallel Strongly Connected Components Computation](https://dl.acm.org/citation.cfm?id=2884048)
