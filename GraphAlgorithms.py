import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import networkx as nx
import random
import logging

random.seed(4)

# BREADTH FIRST SEARCH ALGORITHM
def find_breadthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using breadth first search

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * num_nodes:int - number of nodes in graph
    """
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

    # list of all traversed paths
    path_list = []
    path_list.append(path.copy())

    # list of corresponding visited nodes
    visited_list = []
    visited_list.append(visited)

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)
        path_list.append(path.copy())
        visited_nodes = [i + 1 for i in range(len(visited)) if visited[i] == True]
        visited_list.append(visited_nodes)
        # get node at end of path
        current_node = path[-1]
        
        # if node at end of path is destination, return path found
        if current_node == destination_node:
            print("Valid Path from ", starting_node, "to", destination_node)
            print("Path:", path)
            return True, path_list, visited_list

        # for each not visited neighbor of node at end of path, mark as visited, add to path, and push path to queue
        for neighbor_node in graph.neighbors(current_node):
            if not visited[neighbor_node - 1]:
                visited[neighbor_node - 1] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path)

    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)
    return False, path_list, visited_list

# BREADTH FIRST SEARCH RECURSIVE HELPER
def find_breadthfirstsearch_path_recurse(graph:nx.Graph, starting_node:int, destination_node:int, queue:list[list], visited:list, path_list:list[list], visited_list:list[list]):
    """ recursive function used in find_breadthfirstsearch_path_recursively

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * queue:list[list] - list of paths used in as a queue in search
        * paths:list[list] - list of paths returned
        * visited:list - list of boolean values indicating whether a node is visited
    """
    if len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)
        path_list.append(path.copy())
        visited_nodes = [i + 1 for i in range(len(visited)) if visited[i] == True]
        visited_list.append(visited_nodes)
        # get node at end of path
        current_node = path[-1]
        
        # if node at end of path is destination, return path found
        if current_node == destination_node:
            print("Valid Path from", starting_node, "to", destination_node)
            print("Path:", path)
            return True, path_list, visited_list

        # for each not visited neighbor of node at end of path, mark as visited, add to path, and push path to queue
        for neighbor_node in graph.neighbors(current_node):
            if not visited[neighbor_node - 1]:
                visited[neighbor_node - 1] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path)
        
        return find_breadthfirstsearch_path_recurse(graph=graph, starting_node=starting_node, destination_node=destination_node, queue=queue, visited=visited, path_list=path_list, visited_list=visited_list)
    
    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)
    return False, path_list, visited_list

# BREADTH FIRST SEARCH RECURSIVE ALGORITHM
def find_breadthfirstsearch_path_recursively(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using breadth first search recursive

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * num_nodes:int - number of nodes in graph
    """
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

    # list of all traversed paths
    path_list = []
    path_list.append(path.copy())

    # list of corresponding visited nodes
    visited_list = []
    visited_list.append(visited)

    # call recursive function
    return find_breadthfirstsearch_path_recurse(graph, starting_node, destination_node, queue, visited, path_list, visited_list,)

# DEPTH FIRST SEARCH ALGORITHM
def find_depthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    # final path
    path = []
    path.append(starting_node)

    # list of paths
    stack = []
    stack.append(path)

    # list of visited nodes
    visited = [False for i in range(num_nodes)]
    visited[starting_node - 1] = True

    # list of all traversed paths
    path_list = []
    path_list.append(path.copy())

    # list of corresponding visited nodes
    visited_list = []
    visited_list.append(visited)

    # while stack not empty, check each path in stack
    while len(stack) > 0:
        # set path to path popped from stack
        path = stack.pop()
        path_list.append(path.copy())
        visited_nodes = [i + 1 for i in range(len(visited)) if visited[i] == True]
        visited_list.append(visited_nodes)
        # get node at end of path
        current_node = path[-1]
        
        # if node at end of path is destination, return path found
        if current_node == destination_node:
            print("Valid Path from ", starting_node, "to", destination_node)
            print("Path:", path)
            return True, path_list, visited_list

        # for each not visited neighbor of node at end of path, mark as visited, add to path, and push path to stack
        for neighbor_node in graph.neighbors(current_node):
            if not visited[neighbor_node - 1]:
                visited[neighbor_node - 1] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                stack.append(new_path)

    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)
    return False, path_list, visited_list

            # else:
            #     num_back_track_nodes = len(path) - len(stack[len(stack) - 1]) + 1
            #     new_path = path.copy()
            #     for i in range(num_back_track_nodes):
            #         new_path.pop()
            #         paths.append(new_path)

# GENERATE RANDOM NODE
def generate_random_node(num_nodes:int):
    """ generates random node index for nodes in graph with num_nodes nodes

        * num_nodes:int - number of nodes in graph
    """
    print("Calling Generate Random Node")
    node = random.randrange(start=1, stop=num_nodes + 1)
    print("Random Node:", node)
    return node

# GENERATE NODES
def generate_nodes(num_nodes:int):
    """ generates list of nodes for graph with num_nodes

        * num_nodes:int - number of nodes in graph
    """
    print("Calling Generate Nodes")
    nodes = [i for i in range(1, num_nodes + 1)]
    print("Nodes:", nodes)
    return nodes

# GET MAXIMUM  NUMBER EDGES
def get_maximum_number_edges(num_nodes:int):
    """ finds the maximum number of edges for graph with num_nodes nodes

    * num_nodes:int - number of nodes in graph
    """
    print("Calling Get Maximum Number Edges")
    max_num_edges = num_nodes * (num_nodes - 1) // 2
    print("Maximum Number Edges:", max_num_edges)
    return max_num_edges

# GENERATE RANDOM EDGE
def generate_random_edge(num_nodes:int):
    """ helper function used in generate_random_edges

        * num_nodes:int - number of nodes in graph
    """
    print("Calling Generate Random Edge")
    edge = (random.randrange(1, num_nodes + 1), random.randrange(1, num_nodes + 1))
    print("Random Edge:", edge)
    return edge

# GENERATE RANDOM EDGES
def generate_random_edges(num_nodes:int, num_edges:int):
    """ generates list of random edges for graph with num_nodes nodes and num_edges edges

        * num_nodes:int - number of nodes in graph
        * num_edges:int - number of edges in graph
    """
    print("Calling Generate Random Edges")
    edges = []

    for i in range(num_edges):
        edge = generate_random_edge(num_nodes=num_nodes)

        while (edge in edges) or (edge[0] == edge[1]):
            edge = generate_random_edge(num_nodes=num_nodes)

        edges.append(edge)

    print("Random Edges:", edges)
    return edges

# GENERATE GRAPH
def generate_graph(nodes:list[int], edges:list[tuple]):
    """ generates graph with given nodes and edges

        * nodes:list[int] - list of nodes to add to graph
        * edges:list[tuple] - list of edges to add to graph
    """
    graph = nx.Graph()
    print("Number of Nodes:", len(nodes))
    print("Number of Edges:", len(edges))

    # add nodes [1, number_nodes] to graph
    graph.add_nodes_from(nodes_for_adding=nodes)
    print("Number of Nodes Added:", graph.number_of_nodes())
    print("Nodes in Graph:", graph.nodes())

    # add edges [1, number_edges] to graph
    graph.add_edges_from(ebunch_to_add=edges)
    print("Number of Edges Added:", graph.number_of_edges())
    print("Edges in Graph:", graph.edges())

    return graph

# GENERATE GRAPH ANIMATION
def generate_graph_animation(num_nodes:int, num_edges:int, starting_node:int, destination_node:int, function:callable):
    """ generates graph with num_nodes nodes and num_edges edges, finds path from starting_node to destination_node, and animates process

        * num_nodes:int - number of nodes in graph
        * num_edges:int - number of edges in graph
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
    """
    print("Calling Generate Graph Animation")
    # generate nodes, random edges, and graph
    nodes = generate_nodes(num_nodes=num_nodes)
    print(num_nodes, "Nodes Generated")

    edges = generate_random_edges(num_nodes=num_nodes, num_edges=num_edges)
    print(num_edges, "Edges Generated")

    graph = generate_graph(nodes=nodes, edges=edges)
    print("Graph with", len(nodes), "Nodes and", len(edges), "Edges Generated")
    print("Starting Node:", starting_node)
    print("Destination Node:", destination_node)

    # run search algorithm
    path_found, path_list, visited_list = function(graph=graph, starting_node=starting_node, destination_node=destination_node, num_nodes=num_nodes)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    def update(frame:int):
        ax.clear()

        path_nodes = path_list[frame]
        visited_nodes = visited_list[frame]
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)] if len(path_nodes) > 1 else []

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # start and end node frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=(starting_node, destination_node), node_color="white", edgecolors="black", node_size=450, linewidths=2)

        # visited nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=visited_nodes, node_color="gray", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=path_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black")

        # title
        title = ""
        if function == find_breadthfirstsearch_path or function == find_breadthfirstsearch_path_recursively:
            title = title + "Breadth First Search: "
        # check if path found
        if(frame == len(path_list) - 1):
            if path_found:
                title = title + "Trial " + str(frame) + "- Path Found at " + str(path_nodes)
            else:
                title = title + "Trial " + str(frame) + "- Path Not Found"
        else:
            title = title + "Trial " + str(frame) + "- " + str(path_nodes)
        
        ax.set_title(title)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(path_list), interval=600, repeat=True, repeat_delay=600)
    plt.show()

# set up and call animation function
num_nodes = 16
num_edges = random.randrange(start=0, stop=get_maximum_number_edges(num_nodes) + 1)
starting_node = 1
destination_node = num_nodes

generate_graph_animation(num_nodes=num_nodes, num_edges=num_edges, starting_node=starting_node, destination_node=destination_node, function=find_depthfirstsearch_path)