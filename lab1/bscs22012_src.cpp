#include <algorithm>
#include <iostream>
#include <queue>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

const std::vector<std::vector<int>> input1 = {{1, 2}, {0, 3}, {0, 4}, {1}, {2}};
int startNode							   = 0;
int endNode								   = 4;

const std::unordered_map<char, std::vector<char>> input2 = {
	{'A', {'B', 'C'}}, {'B', {'A', 'D', 'E'}}, {'C', {'A', 'F'}},
	{'D', {'B'}},	   {'E', {'B', 'F'}},	   {'F', {'C', 'E'}}};
const char startNode2 = 'A';
const char endNode2	  = 'F';

template <typename T> void print_vec(std::vector<T> v) {
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
	std::vector<int> traversal;
	std::vector<bool> visited(adjList.size(), false);
	std::queue<int> q;

	visited[startNode] = true;
	q.push(startNode);

	while (!q.empty()) {
		int node = q.front();
		q.pop();
		traversal.push_back(node);

		for (int neighbor : adjList[node]) {
			if (!visited[neighbor]) {
				visited[neighbor] = true;
				q.push(neighbor);
			}
		}
	}

	return traversal;
}

/* You are given an undirected, unweighted graph represented by an
 * adjacency list and two nodes (start_node and end_node).
 * Use BFS to find the shortest path (in terms of the number of edges)
 * between the two nodes. If there is no path, return -1.
 */
int shortest_path_length(std::vector<std::vector<int>> adjList, int startNode,
						 int endNode) {
	std::vector<bool> visited(adjList.size(), false);
	std::queue<std::pair<int, int>> q;

	visited[startNode] = true;
	q.push({startNode, 0});

	while (!q.empty()) {
		std::pair<int, int> node = q.front();
		q.pop();

		if (node.first == endNode)
			return node.second;

		for (int neighbor : adjList[node.first]) {
			if (!visited[neighbor]) {
				visited[neighbor] = true;
				q.push({neighbor, node.second + 1});
			}
		}
	}

	return -1;
}

/*
 * Implement a basic DFS algorithm for a graph represented using
 * an adjacency list. The graph is provided as a dictionary, where
 * the keys are node labels and the values are lists of adjacent nodes.
 */
std::vector<char> dfs(std::unordered_map<char, std::vector<char>> graph,
					  char startNode) {
	std::vector<char> traversal;
	std::unordered_set<char> visited;
	std::stack<char> s;

	s.push(startNode);

	while (!s.empty()) {
		char node = s.top();
		s.pop();

		if (visited.find(node) == visited.end()) {
			visited.emplace(node);
			traversal.push_back(node);

			/* https://en.cppreference.com/w/cpp/ranges/reverse_view */
			for (char x : graph.at(node) | std::views::reverse)
				if (visited.find(x) == visited.end())
					s.push(x);
		}
	}

	return traversal;
}

/*
 * Modify the DFS algorithm to return the path taken from the
 * starting node to a specified target node.
 */
std::vector<char>
dfs_path(const std::unordered_map<char, std::vector<char>> graph,
		 char startNode, char targetNode) {
	std::vector<char> traversal;
	std::unordered_set<char> visited;
	std::unordered_map<char, char> parent;
	std::stack<char> s;

	s.push(startNode);
	parent[startNode] = '\0';

	while (!s.empty()) {
		char node = s.top();
		s.pop();

		if (visited.find(node) == visited.end()) {
			visited.emplace(node);

			if (node == targetNode)
				break;

			for (char x : graph.at(node) | std::views::reverse) {
				if (visited.find(x) == visited.end()) {
					s.push(x);
					parent[x] = node;
				}
			}
		}
	}

	/* reconstruct route */
	for (char cur = targetNode; cur != '\0'; cur = parent[cur])
		traversal.push_back(cur);
	std::reverse(traversal.begin(), traversal.end());

	return traversal;
}

int main() {
	std::cout << "Task 1:\n";
	std::vector<int> task1sol = bfs_traversal(input1, startNode);
	print_vec(task1sol);

	std::cout << "Task 2:\n";
	int task2sol = shortest_path_length(input1, startNode, endNode);
	std::cout << "Path length: " << task2sol << "\n\n";

	std::cout << "Task 3:\n";
	std::vector<char> task3sol = dfs(input2, startNode2);
	print_vec(task3sol);

	std::cout << "Task 4:\n";
	std::vector<char> task4sol = dfs_path(input2, startNode2, endNode2);
	print_vec(task4sol);

	return 0;
}
