// SCC_Hong_Slota.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <iostream>
#include <list>
#include <fstream>
#include <set>
#include <list> 


using namespace std;

// This class represents a directed graph using
// compressed sparse row representation
class Graph
{
	int V;    // No. of vertices
	int E;    // No. of Edges
	int maxColour;

	// Pointer to an array containing adjacency
	// lists
	int* edges;
	int* nodes;
	int* colour;
	int* marked;
	
	list<int> work_queue;

public:
	Graph(int V, int E, char* filename);  // Constructor
	~Graph()
	{
		free(edges);
		free(nodes);
		free(colour);
		free(marked);
	}
	void buildCSRGraph(char filename[]); //Create Graph
	void printInfo();

	int choosePivot(int, int*);
	int checkIndegree(int);
	int checkOutdegree(int);
	int isEdge(int, int);

	void Trim1(); //Remove the 1-SCCs
	void Trim2(); //Remove the 2-SCCs
	void FWBW(int); //Find the SCC
	void WCC(); //Find the individual weakly connected components and add it to the queue

	void SCC(); //Print the SCCs
};

Graph::Graph(int V, int E, char filename[])
{
	this->V = V;
	this->E = E;
	this->maxColour = 0;
	this->nodes = (int *)malloc(sizeof(int)*V);
	this->colour = (int *)calloc(V,sizeof(int));
	this->marked = (int *)calloc(V,sizeof(int));
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
	/*ifstream file;
	file.open(filename);
	if (!file.is_open())
	{
		printf("Could not open file");
		//cerr << "Error: " << strerror_s();

		return;
	}*/
	
	int prev = -1;
	int count_edges = 0;
	//edges[371620] = 0;
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
	//file.close();
	int i;
	/*for (i = 0; i<V; i++)
		printf("Nodes[%d] = %d\n", i, nodes[i]);
	printf("\nEdges   ");
	for (i = 0; i<E; i++)
		printf("%d ", edges[i]);
	printf("\n");*/
	printf("\nDone building CSR !");
}

int Graph::checkIndegree(int i)
{
	if (marked[i]) return -1;
	int j, found = 0;
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

int Graph::checkOutdegree(int i)
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

int Graph::isEdge(int i, int j)
{
	if (i == j) return 0;
	int k, h;
	if (nodes[i] == -1) return 0;
	k = i + 1;
	while (k<V)
	{
		if (nodes[k] != -1)
			break;
		k++;
	}
	for (h = nodes[i]; h<nodes[k]; h++)
	{
		if (edges[h] == j)
			return 1;
	}
	return 0;
}

void Graph::FWBW(int g_colour)
{
	printf("FWBW...\n");
	list<int> queue;
	int *visited1;
	visited1 =(int *)calloc(V,sizeof(int));
	int fw_c = maxColour + 1, bw_c = maxColour + 2, scc_c = maxColour + 3;
	maxColour += 3;
	printf("Colours used: FW:%d BW:%d SCC:%d\n", fw_c, bw_c, scc_c);
	int pivot = 0, i, j;
	choosePivot(g_colour, &pivot);
	if (pivot == -1)
	{
		printf("Pivot was not chosen for colour: %d\n", colour);
	}
	//printf("\nChosen Pivot is %d, of colour: %d\n", pivot, colour[pivot]);
	queue.push_back(pivot);
	int curr;
	int original_c = colour[pivot];

	while (!queue.empty())
	{
		curr = (int)queue.front();
		visited1[curr] = 1;
		queue.pop_front();

		colour[curr] = fw_c;
		int end, start = nodes[curr];
		int end_ind = curr + 1;
		//printf("\n%d %d", start,end_ind);
		while (end_ind < V){
			if (nodes[end_ind] == -1)
				end_ind++;
			else break;
		}
		//printf("\nEnd ind = %d\n",end_ind);
		if (end_ind == V) end = E;
		else end = nodes[end_ind];
		//printf("Iterate from %d to %d for node %d\n", start, end, curr);
		for (i = start; i<end; i++)
		{
			if (colour[edges[i]] == original_c)
			{
				if (!visited1[edges[i]])
					queue.push_back(edges[i]);
				//printf("Pushed: %d\n", edges[i]);
			}
		}
	}
	//BW
	int *visited2;
	visited2 = (int *)calloc(V, sizeof(int));
	queue.push_back(pivot);
	while (!queue.empty())
	{
		curr = (int)queue.front();
		queue.pop_front();
		visited2[curr] = 1;
		//printf("\nPopped: %d\n", curr);
		for (j = 0; j<E; j++)
		{
			if (edges[j] == curr)
			{
				//printf("\nFound edge at position %d\n", j);
				//printf("curr: %d edges[%d]:%d\n", curr, j, edges[j]);
				int min = -1;
				int ind = -1, k;
				for (k = 0; k<V; k++)
				if (nodes[k] > min && nodes[k] <= j){
					min = nodes[k];
					ind = k;
				}
				if (ind == -1) continue;
				if (visited2[ind] == 1) continue;
				//printf("\nExploring edge %d--->%d\n", ind, curr);
				//if (marked[ind]) continue;
				if (colour[ind] == original_c)
				{
					colour[ind] = bw_c;
					queue.push_back(ind);
					//printf("Pushed into BW set: %d\n", ind);
				}
				else if (colour[ind] == fw_c)
				{
					colour[ind] = scc_c;
					marked[ind] = 1;
					queue.push_back(ind);
					//printf("Pushed into SCC set: %d\n", ind);
				}
			}
		}
	}
	for (i = 0; i<V; i++)
	if (colour[i] == scc_c)
		marked[i] = 1;

	colour[pivot] = scc_c;
	marked[pivot] = 1;
}

int Graph::choosePivot(int c, int *pivot)
{
	int i;
	int ret, flag = 0;
	for (i = 0; i<V; i++)
	{
		if (colour[i] == c && marked[i] == 0)
		{
			//printf("\n Color[i]=%d and Marked[i] = %d", colour[i]==c, marked[i]==0);
			*pivot = i;
			return i;
		}
	}
	return -1;
}

void Graph::Trim1()
{
	printf("\nTrim1...\n");
	int i, j, k;
	int change;
	do{
		change = 0;
		for (i = 0; i<V; i++)
		{
			printf("\n Processing Node %d",i);
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
		printf("\n\n\n Done with a round of trimming \n\n\n");
	} while (change);
	printf("MaxColor is %d", maxColour);
}

void Graph::Trim2()
{
	printf("Trim2...\n");
	int i, j;
	for (i = 0; i<V; i++)
	{
		for (j = 0; j<V; j++)
		{
			if (marked[i] || marked[j]) continue;
			if (isEdge(i, j) && isEdge(j, i))
			{

				if ((checkIndegree(i) == 1 && checkIndegree(j) == 1) || (checkOutdegree(i) == 1 && checkOutdegree(j) == 1))
				{
					marked[i] = marked[j] = 1;
					maxColour++;
					colour[i] = colour[j] = maxColour;
				}
			}
		}
	}
}

void Graph::WCC()
{
	printf("WCC...\n");
}

void Graph::printInfo()
{
	int i;
	for (i = 0; i<V; i++)
		printf("\nNodes[%d] = %d", i, nodes[i]);
	printf("\nEdges:");
	for (i = 0; i<E; i++)
		printf("%d ", edges[i]);
	printf("\n");
	for (i = 0; i<V; i++)
	{
		printf("[%d]   Marked:%d   Colour:%d\n", i, marked[i], colour[i]);
	}
	printf("\n");
}

void Graph::SCC()
{
	Trim1();
	//printInfo();
	FWBW(0);

	Trim1();
	Trim2();
	Trim1();

	WCC();
	//printInfo();
}

/*
int main()
{
//Data Filename
char filename[] = "sortedRoadNetwork.txt";
//Number of vertices
int V = 1379917;
//Number of Edges
int E = 1921660;
Graph g(V, E, filename);
g.SCC();
return 0;
}
*/




int main(int argc, char* argv[])
{
	//Data Filename
	char filename[] = "./sortedOnFromRoadNetwork.txt";
	//Number of vertices
	int V = 1393383; //8;
	//Number of Edges
	int E = 3843320; //9; 
	Graph g(V, E, filename);
	g.SCC();
	return 0;
}

