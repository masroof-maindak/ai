# https://docs.python.org/3/library/heapq.html
import heapq

class Graph:
    def __init__(self):
        self.edges = {}
        self.heuristics = {}

    def add_edge(self, src, dst, cost):
        if src not in self.edges:
            self.edges[src] = []
        if dst not in self.edges:
            self.edges[dst] = []
        self.edges[src].append((dst, cost))
        self.edges[dst].append((src, cost))

    def set_heuristic(self, node, h_value):
        self.heuristics[node] = h_value

    def a_star_search(self, src, sink):
        priority_queue = []
        heapq.heappush(priority_queue, (self.heuristics[src], 0, src, []))
        visited = {src: 0}

        while priority_queue:
            estimated_cost, actual_cost, curr_node, path = heapq.heappop(priority_queue)

            # estimated_cost: cost of path + cost of curr_node's heuristic
            # actual_cost:    cost of path

            path = path + [curr_node]

            if curr_node == sink:
                return path, actual_cost

            visited[curr_node] = actual_cost

            for neighbor, cost in self.edges.get(curr_node, []):
                new_actual_cost = actual_cost + cost

                # Going there for the first time OR w/ a better cost than last time
                if new_actual_cost < visited.get(neighbor, float('inf')):

                    new_estimated_cost = new_actual_cost + self.heuristics.get(neighbor, float('inf'))
                    heapq.heappush(priority_queue, (new_estimated_cost, new_actual_cost, neighbor, path))

        return None, float('inf')

def main():
    graph = Graph()

    graph.add_edge('a', 'b', 9)
    graph.add_edge('a', 'c', 4)
    graph.add_edge('a', 'd', 7)
    graph.add_edge('b', 'e', 11)
    graph.add_edge('c', 'e', 17)
    graph.add_edge('c', 'f', 12)
    graph.add_edge('d', 'f', 14)
    graph.add_edge('e', 'z', 5)
    graph.add_edge('f', 'z', 9)

    heuristics = {'a': 21, 'b': 14, 'c': 18, 'd': 18, 'e': 5, 'f': 8, 'z': 0}

    for node, h in heuristics.items():
        graph.set_heuristic(node, h)

    path, cost = graph.a_star_search('a', 'z')

    if path:
        print(f" ath: {' -> '.join(path)} with cost: {cost}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()



