import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random

# BREADTH FIRST SEARCH ALGORITHM
def breadth_first_search(graph, starting_node, destination_node, num_nodes):
    # final path
    path = []
    path.append(starting_node)

    # list of paths
    queue = []
    queue.append(path)

    # list of visited nodes
    visited = [False for i in range(num_nodes)]
    visited[starting_node - 1] = True

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)
        # get node at end of path
        current_node = path[len(path) - 1]
        
        # if node at end of path is destination, return path found
        if current_node == destination_node:
            print(path)
            return path

        # for each not visited neighbor of node at end of path, mark as visited, add to path, and push path to queue
        for neighbor_node in graph.neighbors(current_node):
            if not visited[neighbor_node - 1]:
                visited[neighbor_node - 1] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path)

    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)
    return

# GENERATE RANDOM GRAPH
graph = nx.Graph()
color_map = []

# generate random number of nodes [4, 16]
num_nodes = random.randrange(4, 17)
print("Number of Nodes:", num_nodes)

# generate random number of edges [0, maximum_number_edges]
max_num_edges = num_nodes * (num_nodes - 1) // 2
num_edges = random.randrange(max_num_edges)
print("Number of Edges:", num_edges)

# add nodes [1, number_nodes] to graph
graph.add_nodes_from([i for i in range(1, num_nodes + 1)])
print("Number of Nodes Added:", graph.number_of_nodes())
print("Nodes in Graph:", graph.nodes())

# add edges [1, number_edges] to graph
for i in range(num_edges):
    node_one = random.randrange(num_nodes)
    node_two = random.randrange(num_nodes)

    while (graph.has_edge(node_one, node_two)) or (node_one == node_two):
        node_one = random.randrange(num_nodes + 1)
        node_two = random.randrange(num_nodes + 1)

    graph.add_edge(node_one, node_two)

print("Number of Edges Added:", graph.number_of_edges())
print("Edges in Graph:", graph.edges())

# nx.draw(graph, node_color=color_map)
nx.draw(graph, with_labels=True)
plt.savefig("graph.png")


# generatge random starting node
starting_node = random.randrange(num_nodes) + 1
print("Starting Node:", starting_node)

# generate random destination node
destination_node = random.randrange(num_nodes) + 1
print("Destination Node:", destination_node)

breadth_first_search(graph, starting_node, destination_node, num_nodes)