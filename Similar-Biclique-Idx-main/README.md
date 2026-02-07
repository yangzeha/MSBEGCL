# Identifying Similar-Bicliques in Bipartite Graphs

This project aims to enumerate all maximal similar-bicliques in a bipartite graph. Also, this project proposes algorithms to maintain the index structure when graph updates (i.e., an edge is deleted or a new edge is inserted).

msbe is the executable, and is compiled on Ubuntu 18.04.5, with -O3 optimization.

Folder "datasets" contains an example bipartite graph bi_github. All of our tested bipartite graphs can be obtained from [KONECT](http://konect.cc/networks/github/). 

## Running Format

./msbe [1] input graph  [2] build index flag (0/1)  [3] alpha (0.01～100)  [4] gamma (0.01～1)  [5] index name (LG/GRL3)  [6] load index flag (0/1)  [7] index name (LG/GRL3)  [8] 0/1/2 (i.e., no edge update / edge insertion case / edge deletion case)  [9] number of updated edges  [10] index update algorithm  [11] vertex reduction method  [12] epsilon (similarity constraint)  [13] tau (size constraint)  [14] no similarity constraint on R side

**Running example for building indexGRL3 (with alpha=1, gamma=0.3)**

./msbe ./datasets/bi_github.txt 1 1 0.3 GRL3

**Running example for indexGRL3 based enumeration (with epsilon=0.5, tau=3)**

./msbe ./datasets/bi_github.txt 0 1 0.3 GRL3 1 GRL3 0 1000 heu 4 0.5 3 2

**Running example for index maintenance of 1000 inserted edges**

./msbe ./datasets/bi_github.txt 0 1 0.3 GRL3 1 GRL3 1 1000 heu 4 0.5 3 2

**Running example for index maintenance of 1000 deleted edges**

./msbe ./datasets/bi_github.txt 0 1 0.3 GRL3 1 GRL3 2 1000 heu 4 0.5 3 2

### Note

In the following, I will explain how to set the arguments to execute the code properly. 

In summary, argument[1] is the input graph, arguments[2]-[5] control the index building, arguments[6]-[7] specify which index will be loaded, arguments[8]-[10] specify how to do the index structure maintenance, argument[11] selects vertex reduction method, just set as "4", arguments[12][13] are two important parameters epsilon and tau, argument[14] specifies no similarity constraint on R side, just set as "2".

When argument[2]=1, msbe will build the index according to arguments[3][4][5], other arguments will be ignored. 

After the index is constructed, it is ready to make index based enumeration. Specifically, remember to set argument[2]=0 to invalidate the index build and argument[8]=0 to invalidate the index maintenance. Besides, set argument[6]=1 to load the index.

If you want to test index maintenance, set argument[8]=1 or 2, 1 means edge insertion case, 2 means edge deletion case, argument[9] specifies the number of updated edges (e.g., 1000) and argument[10] specifies index maintenance algorithm, just set as "heu", which corresponds to the index maintenance algorithms in our paper. (Note that, in our implementation, inserted/deleted edges are generated randomly. After updating the graph and the index structure, our program will make the maximal similar-bicliques enumeration on the updated graph with the updated index automatically, this is convenient for us to evaluate the performance of the index maintenance algorithms. Also, our program will record the size of updated index structure automatically.)

Here, GRL3 corresponds to SS in our paper.

## Graph Format

The input graph should be in "binary" format by default. In folder "datasets", there is an example bipartite graph bi_github. Here, edgelist2binary is the executable to transform a "txt" graph into our binary form. 

Our algorithms also support "txt" graph. This needs to comment the function "load_graph_binary" and uncomment the function "load_graph" in main.cpp. 

The txt version should be in the following format:

number of L side vertices \t number of R side vertices \t number of edges \n

v0 \t v1

v0 \t v2

...
