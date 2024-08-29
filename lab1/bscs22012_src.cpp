#include <iostream>
#include <unordered_map>
#include <vector>

const std::vector<std::vector<int>> input1 = {{1, 2}, {0, 3}, {0, 4}, {1}, {2}};
int startNode							   = 0;
int endNode								   = 4;

const std::unordered_map<char, std::vector<char>> input2 = {
	{'A', {'B', 'C'}}, {'B', {'A', 'D', 'E'}}, {'C', {'A', 'F'}},
	{'D', {'B'}},	   {'E', {'B', 'F'}},	   {'F', {'C', 'E'}}};
const char startNode2 = 'A';
const char endNode2	  = 'F';

template <typename T> void printVec(std::vector<T> v) {
	for (auto x : v)
		std::cout << x << ' ';
	std::cout << "\n\n";
}

/*
 * Given an undirected graph represented as an adjacency list,
 * perform a BFS traversal starting from a given start node.
 * Return the list of nodes in the order they are visited.
 */
std::vector<int> bfs_traversal(std::vector<std::vector<int>> adjList,
							   int startNode) {
	return {0};
}

/* You are given an undirected, unweighted graph represented by an
 * adjacency list and two nodes (start_node and end_node).
 * Use BFS to find the shortest path (in terms of the number of edges)
 * between the two nodes. If there is no path, return -1.
 */
int shortest_path_length(std::vector<std::vector<int>> adjList, int startNode,
						 int endNode) {
	return 0;
}

/*
 * Implement a basic DFS algorithm for a graph represented using
 * an adjacency list. The graph is provided as a dictionary, where
 * the keys are node labels and the values are lists of adjacent nodes.
 */
std::vector<int> dfs(const std::unordered_map<char, std::vector<char>> graph,
					 int startNode) {
	return {0};
}

/*
 * Modify the DFS algorithm to return the path taken from the
 * starting node to a specified target node.
 */
std::vector<int>
dfs_path(const std::unordered_map<char, std::vector<char>> graph,
		 char startNode, char targetNode) {
	return {0};
}

int main() {
	std::cout << "Task 1:\n";
	std::vector<int> task1sol = bfs_traversal(input1, startNode);
	printVec(task1sol);

	std::cout << "Task 2:\n";
	int task2sol = shortest_path_length(input1, startNode, endNode);
	std::cout << "Path length: " << task2sol << "\n\n";

	std::cout << "Task 3:\n";
	std::vector<int> task3sol = dfs(input2, startNode2);
	printVec(task3sol);

	std::cout << "Task 4:\n";
	std::vector<int> task4sol = dfs_path(input2, startNode2, endNode2);
	printVec(task4sol);

	return 0;
}
