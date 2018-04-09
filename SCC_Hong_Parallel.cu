#include <iostream>
#include <list>
#include <fstream>
#include <set>
#include <map>
#include <vector>
#include <iterator>
#include <list> 
#include <queue>
#include <chrono>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

//Number of vertices
int V = 15; 	//8297;  //73; 
//Number of Edges
int E = 28; 	//103689;  //100; 

// This class represents a directed graph using compressed sparse row representation
class Graph
{
	public:

	int V;    // No. of vertices
	int E;    // No. of Edges
	int maxColour;
	 
	int* edges;
	int* nodes;
	int* colour;
	int* marked;
	
	Graph(char* filename, int V, int E);  // Constructor
	~Graph()    //Destructor
	{
		free(edges);
		free(nodes);
		free(marked);
		free(colour);
	}
	
	void buildCSRGraph(char filename[]); //Create Graph
	__device__ void printInfo();
	__device__ int checkIndegree(int);
	__device__ int checkOutdegree(int);
	
	//__device__ void Trim1(); //Remove the 1-SCCs
	
};

Graph::Graph(char filename[], int V, int E)
{
	this->V = V;
	this->E = E;
	this->maxColour = 0;
	this->nodes = (int *)malloc(sizeof(int)*V);
	this->colour = (int *)calloc(V, sizeof(int));
	this->marked = (int *)calloc(V, sizeof(int));
	this->edges = (int *)malloc(sizeof(int)*E);
	
	int i;
	for (i = 0; i<V; i++)
	{
		this->nodes[i] = -1;
	}
	buildCSRGraph(filename);
	
}

void Graph::buildCSRGraph(char filename[])
{
	printf("Building CSR...\n");
	int count = 0;
	unsigned int s, d;
	std::ifstream infile(filename);
	
	if (!infile.is_open())
	{
		printf("Could not open Data file\n");
		return;
	}

	int prev = -1;
	int count_edges = 0;

	while (infile >> s >> d)
	{
		count_edges++;
		//printf("%d, %d\n",s, d);
		if (prev == -1 || s != prev)
		{
			nodes[s] = count;
		}
		edges[count] = d;
		count++;
		prev = s;
	}

	infile.close();

	printf("\nDone building CSR!\n");
}

__device__ void Graph::printInfo()
{
	int i;

	//To print nodes and edges arrays
	for (i = 0; i<V; i++)
		printf("\nNodes[%d] = %d", i, nodes[i]);
	printf("\nEdges:");
	for (i = 0; i<E; i++)
		printf("%d ", edges[i]);
	printf("\n");
}

__device__ int Graph::checkIndegree(int i)
{
	if (marked[i]) return -1;
	int j;
	int validInEdges = 0;
	for (j = 0; j<E; j++)
	{
		if (edges[j] == i)
		{
			int min = -1;
			int ind = -1, k;
			for (k = 0; k<V; k++)
			if (nodes[k] > min && nodes[k] <= j){
				min = nodes[k];
				ind = k;
			}
			if (marked[ind]) continue;
			validInEdges++;
		}
	}
	return validInEdges;
}

__device__ int Graph::checkOutdegree(int i)
{
	if (marked[i]) return -1;
	if (nodes[i] == -1) return 0;
	int k, h;
	k = i + 1;
	int end;
	while (k<V)
	{
		if (nodes[k] != -1)
			break;
		k++;
	}
	int validOutEdges = 0;
	end = nodes[k];
	if (k == V) end = E;
	for (h = nodes[i]; h<end; h++)
	{
		if (marked[edges[h]] == 0)
			validOutEdges++;
	}
	return validOutEdges;
}

/*__device__ void Graph::Trim1()
{
	printf("\nTrim1...\n");
	int i, j, k;
	int change;
	do{
		change = 0;
		for (i = 0; i<V; i++)
		{
			//printf("\n Processing Node %d", i);
			if (checkOutdegree(i) == 0)
			{
				//printf("\nOutdegree is zero for %d\n",i);
				marked[i] = 1;
				change = 1;
				maxColour++;
				colour[i] = maxColour;
				continue;
			}
			else if (checkIndegree(i) == 0)
			{
				//printf("\nIndegree is zero for %d\n",i);
				marked[i] = 1;
				change = 1;
				maxColour++;
				colour[i] = maxColour;
			}
		}
		//printf("\n\n\n Done with a round of trimming \n\n\n");
	} while (change);
	printf("\nMaxColor is %d", maxColour);
}*/

__global__ void Trim1(Graph* d_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= d_g->V )
	   return;

	//printf("\n Processing Node %d", i);
	if (d_g->checkOutdegree(i) == 0)
	{
		//printf("\nOutdegree is zero for %d\n",i);
		d_g->marked[i] = 1;
		d_g->maxColour++;
		d_g->colour[i] = d_g->maxColour;
		return;
	}
	else if (d_g->checkIndegree(i) == 0)
	{
		//printf("\nIndegree is zero for %d\n",i);
		d_g->marked[i] = 1;
		d_g->maxColour++;
		d_g->colour[i] = d_g->maxColour;
	}
   
}

void SCC(Graph* d_g)
{
	//Trim 1
	int threads, blocks;
	threads = 1024;
	blocks = V/threads + 1;
	Trim1<<<blocks, threads >>>(d_g);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
	//Data Filename
	char filename[] = "./smallDummyDataSorted.txt";

	Graph h_g(filename, V, E);

	Graph *d_g;
	cudaMalloc((void **)&d_g, sizeof(Graph));
	cudaMemcpy(d_g, &h_g, sizeof(Graph), cudaMemcpyHostToDevice);

	int *h_edges, *h_nodes, *h_marked, *h_colour;
    cudaMalloc((void **)&h_edges, sizeof(int)*E);
	cudaMemcpy(h_edges, h_g.edges, sizeof(int)*E, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&h_nodes, sizeof(int)*V);
	cudaMemcpy(h_nodes, h_g.nodes, sizeof(int)*V, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&h_marked, sizeof(int)*V);
	cudaMemcpy(h_marked, h_g.marked, sizeof(int)*V, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&h_colour, sizeof(int)*V);
    cudaMemcpy(h_colour, h_g.colour, sizeof(int)*V, cudaMemcpyHostToDevice);
    
	cudaMemcpy(&(d_g->edges), &h_edges, sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_g->nodes), &h_nodes, sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_g->marked), &h_marked, sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_g->colour), &h_colour, sizeof(int *), cudaMemcpyHostToDevice);

	SCC(d_g);
	
	return 0;
}
