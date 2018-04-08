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
	//int checkIndegree(int);
	//int checkOutdegree(int);
	
	void Trim1(); //Remove the 1-SCCs
	

	void SCC(); //Print the SCCs
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

__global__ void useClass(Graph* d_g)
{
   
	d_g->printInfo();
};

/*int Graph::checkIndegree(int i)
{
	if (d_marked[i]) return -1;
	int j; // found = 0;
	int validInEdges = 0;
	for (j = 0; j<E; j++)
	{
		if (d_edges[j] == i)
		{
			int min = -1;
			int ind = -1, k;
			for (k = 0; k<V; k++)
			if (d_nodes[k] > min && d_nodes[k] <= j)
			{
				min = d_nodes[k];
				ind = k;
			}
			if (d_marked[ind]) continue;
			validInEdges++;
		}
	}
	return validInEdges;
}

int Graph::checkOutdegree(int i)
{
	if (d_marked[i]) return -1;
	if (d_nodes[i] == -1) return 0;
	int k, h;
	k = i + 1;
	int end;
	while (k<V)
	{
		if (d_nodes[k] != -1)
			break;
		k++;
	}
	int validOutEdges = 0;
	end = d_nodes[k];
	if (k == V) end = E;
	for (h = d_nodes[i]; h<end; h++)
	{
		if (d_marked[d_edges[h]] == 0)
			validOutEdges++;
	}
	return validOutEdges;
}


__global__ void Trim1Kernel(int* d_nodes, int* d_edges, int* d_marked, int* d_colour, int* d_maxColour)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if (checkOutdegree(i) == 0)
	{
		//printf("\nOutdegree is zero for %d\n",i);
		d_marked[i] = 1;
		//change = 1;
		*d_maxColour++;
		d_colour[i] = *d_maxColour;
		return;
	}
	else if (checkIndegree(i) == 0)
	{
		//printf("\nIndegree is zero for %d\n",i);
		d_marked[i] = 1;
		//change = 1;
		*d_maxColour++;
		d_colour[i] = *d_maxColour;
	}

}*/

void Graph::Trim1()
{
	printf("\nTrim1...\n");
	/*int blocks, threads;
	threads = 1024;
	blocks = V/threads + 1;

	Trim1Kernel<<<blocks,threads>>>(d_nodes, d_edges, d_marked, d_colour, d_maxColour);
	/*int i; // j, k;
	int change;
	do{
		change = 0;
		for (i = 0; i<V; i++)
		{
			//printf("\n Processing Node %d", i);
			if (checkOutdegree(i) == 0)
			{
				//printf("\nOutdegree is zero for %d\n",i);
				h_marked[i] = 1;
				change = 1;
				h_maxColour++;
				h_colour[i] = h_maxColour;
				continue;
			}
			else if (checkIndegree(i) == 0)
			{
				//printf("\nIndegree is zero for %d\n",i);
				h_marked[i] = 1;
				change = 1;
				h_maxColour++;
				h_colour[i] = h_maxColour;
			}
		}
		//printf("\n\n\n Done with a round of trimming \n\n\n");
	} while (change);
	printf("MaxColor is %d", h_maxColour);*/
}


void Graph::SCC()
{
	Trim1();
}

int main(int argc, char* argv[])
{
	//Data Filename
	char filename[] = "./smallDummyDataSorted.txt";

	//Number of vertices
	int V = 15; 	//8297;  //73; 
	//Number of Edges
	int E = 28; 	//103689;  //100; 

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

    useClass<<<1,1>>>(d_g);
    cudaDeviceSynchronize();
	
	return 0;
}
