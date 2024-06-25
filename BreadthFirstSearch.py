import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random
import logging

random.seed(8)

# BREADTH FIRST SEARCH ALGORITHM
def find_path(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    print("Calling Iterative Breadth First Search")
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
            print("Valid Path from ", starting_node, "to", destination_node)
            print("Path:", path)
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

def find_path_recurse(graph:nx.Graph, starting_node:int, destination_node:int, queue:list[list], visited:list):
    if len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)
        # get node at end of path
        current_node = path[len(path) - 1]
        
        # if node at end of path is destination, return path found
        if current_node == destination_node:
            print("Valid Path from", starting_node, "to", destination_node)
            print("Path:", path)
            return path

        # for each not visited neighbor of node at end of path, mark as visited, add to path, and push path to queue
        for neighbor_node in graph.neighbors(current_node):
            if not visited[neighbor_node - 1]:
                visited[neighbor_node - 1] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path)
        
        return find_path_recurse(graph, starting_node, destination_node, queue, visited)
    
    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)
    return

# BREADTH FIRST SEARCH RECURSIVE ALGORITHM
def find_path_recursively(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    print("Calling Recursive Breadth First Search")
    # final path
    path = []
    path.append(starting_node)

    # list of paths
    queue = []
    queue.append(path)

    # list of visited nodes
    visited = [False for i in range(num_nodes)]
    visited[starting_node - 1] = True

    # call recursive function
    return find_path_recurse(graph, starting_node, destination_node, queue, visited)

def generate_random_node(num_nodes:int):
    print("Calling Generate Random Node")
    node = random.randrange(0, num_nodes) + 1
    print("Random Node:", node)
    return node

def generate_nodes(num_nodes:int):
    print("Calling Generate Nodes")
    nodes = [i for i in range(1, num_nodes + 1)]
    print("Nodes:", nodes)
    return nodes

def get_maximum_number_edges(num_nodes:int):
    print("Calling Get Maximum Number Edges")
    max_num_edges = num_nodes * (num_nodes - 1) // 2
    print("Maximum Number Edges:", max_num_edges)
    return max_num_edges

def generate_random_edge(num_nodes:int):
    print("Calling Generate Random Edge")
    edge = (random.randrange(num_nodes), random.randrange(num_nodes))
    print("Random Edge:", edge)
    return edge

def generate_random_edges(num_nodes:int, num_edges:int):
    print("Calling Generate Random Edges")
    edges = []

    for i in range(num_edges):
        edge = generate_random_edge(num_nodes)

        while (edge in edges) or (edge[0] == edge[1]):
            edge = generate_random_edge(num_nodes)

        edges.append(edge)

    print("Random Edges:", edges)
    return edges

# GENERATE GRAPH
def generate_graph(nodes:list[int], edges:list[tuple]):
    graph = nx.Graph()
    print("Number of Nodes:", len(nodes))
    print("Number of Edges:", len(edges))

    # add nodes [1, number_nodes] to graph
    graph.add_nodes_from(nodes)
    print("Number of Nodes Added:", graph.number_of_nodes())
    print("Nodes in Graph:", graph.nodes())

    # add edges [1, number_edges] to graph
    graph.add_edges_from(edges)
    print("Number of Edges Added:", graph.number_of_edges())
    print("Edges in Graph:", graph.edges())

    return graph

# generate nodes
num_nodes = 10
nodes = generate_nodes(10)

# generate random number of edges [0, maximum_number_edges]
num_edges = random.randrange(0, get_maximum_number_edges(num_nodes) + 1)
edges = generate_random_edges(num_nodes, num_edges)

# generatge random starting node
starting_node = generate_random_node(num_nodes)
print("Starting Node:", starting_node)

# generate random destination node
destination_node = generate_random_node(num_nodes)
print("Destination Node:", destination_node)

# generate random graph
graph = generate_graph(nodes, edges)
nx.draw(graph, with_labels=True)
plt.savefig("graph.png")
# nx.draw(graph, node_color=color_map)

# test breadth first search
find_path(graph, starting_node, destination_node, num_nodes)

# test breadth first search recursive
find_path_recursively(graph, starting_node, destination_node, num_nodes)