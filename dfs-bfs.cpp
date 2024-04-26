#include <omp.h>
#include <chrono>
#include <iostream>
#include <queue>
#include <vector>

#define NUM_THREADS 6

using namespace std;
using namespace chrono;

struct Node {
	int id;
	vector<int> adj;
};

// Function to perform sequential BFS traversal
void sequentialBFS(const vector<Node>& graph, int start) {
	queue<int> queue;
	vector<bool> visited(graph.size(), false);

	queue.push(start);
	visited[start] = true;

	auto start_time = high_resolution_clock::now();

	while (!queue.empty()) {
		int current = queue.front();
		queue.pop();

		for (int neighbor : graph[current].adj) {
			if (!visited[neighbor]) {
				queue.push(neighbor);
				visited[neighbor] = true;
			}
		}
	}

	auto end_time = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end_time - start_time);

	cout << "Sequential BFS: ";
	cout << duration.count() << " milliseconds" << endl;
}

// Parallel BFS traversal using OpenMP
void parallelBFS(const vector<Node>& graph, int start) {
	queue<int> queue;
	vector<bool> visited(graph.size(), false);

	queue.push(start);
	visited[start] = true;

	auto start_time = high_resolution_clock::now();

	#pragma omp parallel // num_threads(NUM_THREADS)
	while (!queue.empty()) {
		#pragma omp for nowait	// Avoid unnecessary synchronization
		for (int i = 0; i < queue.size(); ++i) {
			int current = queue.front();
			queue.pop();

			#pragma omp critical  // Critical section for shared queue access
			{
				for (int neighbor : graph[current].adj) {
					if (!visited[neighbor]) {
						queue.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
	}

	auto end_time = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end_time - start_time);

	cout << "Parallel BFS: ";
	cout << duration.count() << " milliseconds" << endl;
}

// Internal Recursive Function to perform sequential DFS traversal
void sequentialDFSInternal(const vector<Node>& graph, int start, vector<bool>& visited) {
	visited[start] = true;
	for (int neighbor : graph[start].adj) {
		if (!visited[neighbor]) {
			sequentialDFSInternal(graph, neighbor, visited);
		}
	}
}

// Function to perform sequential DFS traversal
void sequentialDFS(const vector<Node>& graph, int start) {
	vector<bool> visited(graph.size(), false);

	auto start_time = high_resolution_clock::now();

	sequentialDFSInternal(graph, start, visited);

	auto end_time = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end_time - start_time);
	
	cout << "Sequential DFS: ";
	cout << duration.count() << " milliseconds" << endl;
}

// Parallel DFS traversal using OpenMP
void parallelDFS(const vector<Node>& graph, int start) {
	vector<bool> visited(graph.size(), false);

	auto start_time = high_resolution_clock::now();

	#pragma omp parallel //num_threads(NUM_THREADS)
	{
		#pragma omp single	// Only one thread executes this block
		sequentialDFSInternal(graph, start, visited);
	}

	auto end_time = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end_time - start_time);

	cout << "Parallel DFS: ";
	cout << duration.count() << " milliseconds" << endl;
}

int main() {
	int size;
	cout << "Enter size of graph (no. of nodes): ";
	cin >> size;

	vector<Node> graph(size);

	for (int i = 0; i < size; i++) {
		graph[i].adj = {};
		for (int j = 0; j < size; j++) {
			graph[i].adj.push_back(j);
		}
	}

	int start = 0;

	sequentialBFS(graph, start);
	sequentialDFS(graph, start);
	parallelBFS(graph, start);
	parallelDFS(graph, start);
}