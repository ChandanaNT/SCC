// SCC_Hong_Slota.cpp : Defines the entry point for the console application.

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
	int* inverted_edges;
	int* inverted_nodes;

	queue<int> work_queue;
	map <int, std::vector<int> > colourMap;

public:
	Graph(int V, int E, char* filename);  // Constructor
	~Graph()    //Destructor
	{
		free(edges);
		free(nodes);
		free(inverted_edges);
		free(inverted_nodes);
		free(colour);
		free(marked);
	}
	void DFS(int index, int * visited, int colour, int old_colour);
	void buildCSRGraph(char filename[]); //Create Graph
	void buildCSRInverseGraph(char filename[]); //Creates Inverse Graph
	void buildColourMap();
	void printColourMap();
	void printInfo();
	void findBiggestSCC();

	int choosePivot(int, int*);
	int checkIndegree(int);
	int checkOutdegree(int);
	int isEdge(int, int);

	void Trim1(); //Remove the 1-SCCs
	void Trim2(); //Remove the 2-SCCs
	void FWBW(int); //Find the SCC
	void WCC(); //Find the individual weakly connected components and add it to the queue
	void repeated_FWBW();
	void FWBW_with_queue(int pivot);

	void SCC(); //Print the SCCs
};

Graph::Graph(int V, int E, char filename[])
{
	this->V = V;
	this->E = E;
	this->maxColour = 0;
	this->nodes = (int *)malloc(sizeof(int)*V);
	this->colour = (int *)calloc(V, sizeof(int));
	this->marked = (int *)calloc(V, sizeof(int));
	this->edges = (int *)malloc(sizeof(int)*E);
	this->inverted_edges = (int *)malloc(sizeof(int)*E);
	this->inverted_nodes = (int *)malloc(sizeof(int)*V);
	int i;
	for (i = 0; i<V; i++)
	{
		this->nodes[i] = -1;
		this->inverted_nodes[i] = -1;
	}
	buildCSRGraph(filename);
	buildCSRInverseGraph(filename);
	printInfo();
}

void Graph::buildCSRGraph(char filename[])
{
	printf("Building CSR...\n");
	int count = 0;
	unsigned int s, d;
	std::ifstream infile(filename);
	/*ifstream file;
	if (!file.is_open())
	{
	printf("Could not open file");
	return;
	}*/

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
	int i;
	/*for (i = 0; i<V; i++)
	printf("Nodes[%d] = %d\n", i, nodes[i]);
	printf("\nEdges   ");
	for (i = 0; i<E; i++)
	printf("%d ", edges[i]);
	printf("\n");*/
	printf("\nDone building CSR!\n");
}

void Graph::buildCSRInverseGraph(char filename[])
{
	printf("\nBuilding Inverse CSR...\n");
	int l = strlen(filename);
	printf("Length of filename is: %d\n",l);
	char *inverseFilename = (char *)malloc(l*sizeof(char));
	int i;
	char copytext[] = "_inv.txt";
	for (i = 0; i < l; i++)
		inverseFilename[i] = filename[i];
	for (i = 0; i <= 8; i++)
		inverseFilename[i + l - 4] = copytext[i];
	printf("File to be opened is:||%s||", inverseFilename);
	int count = 0;
	unsigned int s, d;
	std::ifstream infile(inverseFilename); 
	ifstream file;
	file.open(filename);
	if (!file.is_open())
	{
		printf("Could not open file");
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
			inverted_nodes[s] = count;
		}
		inverted_edges[count] = d;
		count++;
		prev = s;
	}

	infile.close();

	/*for (i = 0; i<V; i++)
	printf("Nodes[%d] = %d\n", i, nodes[i]);
	printf("\nEdges   ");
	for (i = 0; i<E; i++)
	printf("%d ", edges[i]);
	printf("\n");*/
	printf("\nDone building Inversted CSR!\n");
}

void Graph::buildColourMap()
{
	int i;
	for (i = 0; i < maxColour; i++)
	{
		vector<int> nodeList;
		colourMap.insert(pair <int, std::vector<int> >(i, nodeList));
	}
	for (i = 0; i < V; i++)
	{
		colourMap[colour[i]].push_back(i);
	}
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
	visited1 = (int *)calloc(V, sizeof(int));
	int fw_c = maxColour + 1, bw_c = maxColour + 2, scc_c = maxColour + 3;
	maxColour += 3;
	printf("Colours used: FW:%d BW:%d SCC:%d\n", fw_c, bw_c, scc_c);
	int pivot = 0, i, j;
	choosePivot(g_colour, &pivot);
	if (pivot == -1)
	{
		printf("Pivot was not chosen for colour: %d\n", colour);
		return ;
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
			if (colour[edges[i]] == original_c && marked[edges[i]] == 0)
			{
				if (!visited1[edges[i]])
				{
					visited1[edges[i]] = 1;
					queue.push_back(edges[i]);
				}
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
				if (visited2[ind] == 1 || marked[ind] == 1) continue;
				//printf("\nExploring edge %d--->%d\n", ind, curr);
				//if (marked[ind]) continue;
				if (colour[ind] == original_c)
				{
					colour[ind] = bw_c;
					visited2[ind] = 1;
					queue.push_back(ind);
					//printf("Pushed into BW set: %d\n", ind);
				}
				else if (colour[ind] == fw_c)
				{
					colour[ind] = scc_c;
					marked[ind] = 1;
					visited2[ind] = 1;
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
	*pivot = -1;
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
	printf("MaxColor is %d", maxColour);
}

void Graph::Trim2()
{
	printf("\nTrim2...\n");
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

void Graph::DFS(int index, int * visited, int c, int old_colour)
{
	//printf("DFS: Processing node: %d in colour group: %d\n", index, old_colour);
	int i;
	visited[index] = 1;
	colour[index] = c;
	int min_ind, max_ind;
	if (nodes[index] != -1)
	{
	min_ind = nodes[index];

	if (index + 1 == V) max_ind = E;
	else max_ind = nodes[index + 1];
	for (i = min_ind; i < max_ind; i++)
	{
		if (visited[i] == 0 && marked[i] == 0 && colour[i] == old_colour)
		{
			visited[i] = 1;
			colour[i] = c;
			DFS(edges[i], visited, c, old_colour);
		}
	}
}
	
	if (inverted_nodes[index] != -1){
		min_ind = inverted_nodes[index];
		if (index + 1 == V) max_ind = E;
		else max_ind = inverted_nodes[index + 1];

		for (i = min_ind; i < max_ind; i++)
		{
			if (visited[i] == 0 && marked[i] == 0 && colour[i] == old_colour)
			{   
				visited[i] = 1;
			    colour[i] = c;
				DFS(edges[i], visited, c, old_colour);
			}
		}
	}

}

void Graph::WCC()
{
	printf("\nWCC...\n");
	int *visited = (int *)calloc(V,sizeof(int));
	int i;
	for (i = 0; i < V;i++)
	if (marked[i] == 0)
	{
		if (visited[i] == 0)
		{
			printf("Performing DFS from node: %d\n\n",i);
			maxColour++;
			Graph::DFS(i, visited, maxColour, colour[i]);
			work_queue.push(maxColour);
		}		
	}
	
	printf("\nWCC Done...\n");
}

void Graph::repeated_FWBW(){
	while (!work_queue.empty()){
		int q = (int)work_queue.front();
		work_queue.pop();
		FWBW_with_queue(q);
	}
}

void Graph::FWBW_with_queue(int g_colour)
{
	printf("\nFWBW with Queue...\n");
	list<int> queue;
	int *visited1;
	visited1 = (int *)calloc(V, sizeof(int));
	int fw_c = maxColour + 1, bw_c = maxColour + 2, scc_c = maxColour + 3;
	maxColour += 3;
	printf("Colours used: FW:%d BW:%d SCC:%d\n", fw_c, bw_c, scc_c);
	int pivot = 0, i, j;
	choosePivot(g_colour, &pivot);
	if (pivot == -1)
	{
		printf("Pivot was not chosen for colour: %d\n", g_colour);
		return;
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
			if (colour[edges[i]] == original_c && marked[i] == 0)
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
		visited2[curr] = 1;
		queue.pop_front();
		
		//if (colour[curr] == fw_c)
			//colour[curr] == scc_c;
		int end, start = inverted_nodes[curr];
		int end_ind = curr + 1;
		//printf("\n%d %d", start,end_ind);
		while (end_ind < V){
			if (inverted_nodes[end_ind] == -1)
				end_ind++;
			else break;
		}
		//printf("\nEnd ind = %d\n",end_ind);
		if (end_ind == V) end = E;
		else end = inverted_nodes[end_ind];
		//printf("Iterate from %d to %d for node %d\n", start, end, curr);
		for (i = start; i<end; i++)
		{
			int v = inverted_edges[i];
			if (visited2[v] == 0 && marked[v] == 0)
			{
				if (colour[v] == original_c)
				{
					colour[v] = bw_c;
					visited2[v] = 1;
					queue.push_back(v);
					//printf("Pushed into BW set: %d\n", ind);
				}
				else if (colour[v] == fw_c)
				{
					colour[v] = scc_c;
					marked[v] = 1;
					visited2[v] = 1;
					queue.push_back(v);
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

	work_queue.push(bw_c);
	work_queue.push(fw_c);

}

void Graph::printInfo()
{
	int i, j;
	int marked_count = 0;

	//To print nodes and edges arrays
	for (i = 0; i<V; i++)
		printf("\nNodes[%d] = %d", i, nodes[i]);
	printf("\nEdges:");
	for (i = 0; i<E; i++)
		printf("%d ", edges[i]);
	printf("\n");



	for (i = 0; i<V; i++)
		printf("\ninverted_Nodes[%d] = %d", i, inverted_nodes[i]);
	printf("\nInverted_Edges:");
	for (i = 0; i<E; i++)
		printf("%d ", inverted_edges[i]);
	printf("\n");

	for (i = 0; i<V; i++)
	{
		printf("[%d]   Marked:%d   Colour:%d\n", i, marked[i], colour[i]);
		if (marked[i])
			marked_count++;

	}
	printf("\nNumber of marked nodes are %d", marked_count);

	for (i = 0; i <= maxColour; ++i)
	{
		printf("\nNodes belonging to colour %d are  ", i);
		for (j = 0; j < V; j++)
		{
			if (colour[j] == i)
				printf("%d ", j);

		}
	}
	printf("\n");
}

void Graph::printColourMap()
{
	map <int, std::vector<int> > ::iterator itr;
	vector<int>::iterator jtr;
	cout << "\nThe hash map is \n";
	for (itr = colourMap.begin(); itr != colourMap.end(); ++itr)
	{
		cout << '\t' << itr->first << '\t';
		for (jtr = itr->second.begin(); jtr != itr->second.end(); jtr++)
		{
			cout << *jtr << " ";
		}
		cout << "\n";
	}
}

void Graph::findBiggestSCC()
{
	int max_size = -1, size;
	map <int, std::vector<int> > ::iterator itr;
	vector<int>::iterator jtr;
	for (itr = colourMap.begin(); itr != colourMap.end(); ++itr)
	{
		size = itr->second.size();
		if(max_size < size)
		   max_size = size;		
	}
	printf("\nSize of the biggest SCC is %d ", max_size);
}

void Graph::SCC()
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //Hong Algorithm to find SCCs
	Trim1();
	FWBW(0);
	//printInfo();

	Trim1();
	Trim2();
	Trim1();

	WCC();
	repeated_FWBW();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    double duration = duration_cast<microseconds>( t2 - t1 ).count();


	buildColourMap();
	printColourMap();
	findBiggestSCC();
	//printInfo();
    
    printf("\nExecution time of SCC algorithm : %lf microseconds \n", duration);
         	
}


int main(int argc, char* argv[])
{
	//Data Filename
	char filename[] = "./smallDummyDataSorted.txt";
	//Number of vertices
	int V = 15; //8297;  //73; 
	//Number of Edges
	int E = 28; //103689;  //100; 
	Graph g(V, E, filename);
	g.SCC();
	return 0;
}

