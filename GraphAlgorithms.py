import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random
import logging

random.seed(4)

# UPDATE SEARCH ALGORITHM GRAPHICS
def update_search_algorithm_graphics(traversed_list:list[list], visited_list:list[list], labels_list:list[str], path:list[int], visited:list[int], label:str):
    """ updates graphical information for search algorithms

        * traversed_list:list[list] - list of lists of traversed nodes
        * visited_list:list[list] - list of lists of visited nodes
        * labels_list:list[str] - list of strings of labels
        * path:list[int] - traversed nodes list to be added to traversed_list
        * visited:list[int] - visited nodes list to be added to visited_list
        * label:str - label to be added to labels

    """
    print("Calling Update Search Algorithm Graphics")

    traversed_list.append(path.copy())
    visited_list.append(visited.copy())
    labels_list.append(label)

# BREADTH FIRST SEARCH ALGORITHM
def find_breadthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int):
    """ finds path in the graph between the starting node and the destination node using breadth first search
        returns title, traversed_list, visited_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node

    """
    print("Calling Iterative Breadth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    queue = []
    queue.append(path.copy())

    # list of visited nodes
    visited = []
    visited.append(starting_node)

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    visited_list = []
    labels_list = []
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)

        # get node at end of path
        current_node = path[-1]

        # graphics
        label = "Checking Neighbors of Node " + str(current_node)
        update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        # check each neighbor
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_nodes = []
        for node in neighbors:
            if node not in visited:
                neighbor_nodes.append(node)

        # graphics
        if len(neighbor_nodes) == 0:
            label = "No Neighbors Found"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        else:
            for neighbor_node in neighbor_nodes:
                # for each not visited neighbor, mark as visited, add to path, and push path to queue
                visited.append(neighbor_node)
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path.copy())

                # graphics
                label = "Neighbor Node " + str(neighbor_node) + " Visited"
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                # if node at end of path is destination, return path found
                if neighbor_node == destination_node:
                    print("Valid Path From ", starting_node, "to", destination_node)
                    print("Path:", new_path)

                    # graphics
                    label = "Path Found at " + str(new_path)
                    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                    return "Breadth First Search: ", traversed_list, visited_list, labels_list
                    
                # graphics
                label = "Checking Neighbors of Node " + str(current_node)
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

                # graphics
                if neighbor_node == neighbor_nodes[len(neighbor_nodes) - 1]:
                    label = "No Neighbors Found"
                    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # if destination not found, return path not found 
    print("No Valid Path From", starting_node, "to", destination_node)

    # graphics
    label = "No Path Found"
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    return "Breadth First Search: ", traversed_list, visited_list, labels_list

# DEPTH FIRST SEARCH ALGORITHM
def find_depthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int):
    """ finds path in the graph between the starting node and the destination node using depth first search
        returns title, traversed_list, visited_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node

    """
    print("Calling Depth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    stack = []
    stack.append(path.copy())

    # list of visited nodes
    visited = []
    visited.append(starting_node)

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    visited_list = []
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # while stack not empty, check each path in queue
    while len(stack) > 0:
        # set path to path from stack
        path = stack[-1]

        # get node at end of path
        current_node = path[-1]

        # graphics
        label = "Checking Neighbors of Node " + str(current_node)
        update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

        # check if unvisited neighbor exists
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_node = None
        for node in neighbors:
            if node not in visited:
                neighbor_node = node
                break

        # for the first not visited neighbor, mark as visited, add to path, and push path to stack
        if neighbor_node:
            visited.append(neighbor_node)
            new_path = path.copy()
            new_path.append(neighbor_node)
            stack.append(new_path.copy())

            # graphics
            label = "Neighbor Node " + str(neighbor_node) + " Visited"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

            # if node at end of path is destination, return path found
            if neighbor_node == destination_node:
                print("Valid Path From ", starting_node, "to", destination_node)
                print("Path:", new_path)

                # graphics
                label = "Path Found at " + str(new_path)
                update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=new_path, visited=visited, label=label)

                return "Depth First Search: ", traversed_list, visited_list, labels_list
            
        else:
            # graphics
            path = stack.pop(-1)
            label = "No Neighbors Found"
            update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    # if destination not found, return path not found 
    print("No Valid Path From", starting_node, "to", destination_node)

    # graphics
    label = "No Path Found"
    update_search_algorithm_graphics(traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list, path=path, visited=visited, label=label)

    return "Depth First Search: ", traversed_list, visited_list, labels_list

# UPDATE SSSP ALGORITHM GRAPHICS
def update_sssp_algorithm_graphics(sssp_list:list[list], neighbors_list:list[list], distances_list:list[dict], edges_list:list[list], labels_list:list[str], sssp:list[int], neighbors:list[int], distances:dict, edges:list[tuple], label:str):
    """ updates graphical information for search algorithms

        * sssp_list:list[list] - list of lists of sssp nodes
        * neighbors_list:list[list] - list of list of neighbor nodes
        * distances_list:list[dict] - list of dict of distances from starting node
        * edges_list:list[list] - list of list of edge nodes
        * labels_list:list[str] - list of strings of labels
        * sssp:list[int] - sssp nodes list to be added to sssp_list
        * neighbors:list[int] - neighbor nodes list to be added to neighbors_list
        * distances:dict - distances dict to be formatted and added to distances_list
        * edges:list[tuple] - edge tuples list to be added to edges_list
        * label:str - label to be added to labels

    """
    print("Calling Update Search Algorithm Graphics")
    distances_formatted = distances.copy()
    for i in range(len(distances_formatted)):
        distances_formatted[i] = str(i) + ": " + str(distances[i])
    
    sssp_list.append(sssp.copy())
    neighbors_list.append(neighbors.copy())
    distances_list.append(distances_formatted.copy())
    edges_list.append(edges.copy())
    labels_list.append(label)

# GET SSSP NEIGHBORS
def get_sssp_neighbors(graph:nx.Graph, sssp:list[int]):
    """ updates graphical information for search algorithms

        * graph:nx.Graph - graph to find path
        * sssp:list[int] - sssp nodes list

    """
    print("Calling Update Search Algorithm Graphics")

    neighbors = []

    for node in sssp:
        for neighbor in graph.neighbors(node):
            if neighbor not in neighbors and neighbor not in sssp:
                neighbors.append(neighbor)
    
    return neighbors

# DIJKSTRA'S ALGORITHM
def find_dijkstra_path(graph:nx.Graph, starting_node:int, num_nodes:int):
    """ finds shortest path in the graph between all nodes using dikstra's algorithm
        returns title, sssp_list, neighbors_list, edges_list, labels_list used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the first graph node
        * num_nodes:int - number of nodes in graph

    """
    print("Calling Dijkstra's Algorithm")

    # final sssp
    sssp = []
    edges = []
    neighbors = []

    # list of distances from starting node to all nodes
    distances = {i:np.inf for i in range(0, num_nodes)}
    distances[starting_node] = 0

    # list of all sssp nodes, sssp edges, neighbor nodes, and labels for graphics
    sssp_list = []
    neighbors_list = []
    distances_list = []
    edges_list = []
    label = "Finding SSSP From Node " + str(starting_node)
    labels_list = []
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, label=label)

    # local memory used to help graphics
    minimum_distance_neighbor_node = starting_node
    minimum_distance_sssp_node = starting_node
    minimum_distance = 0

    # get neighbors of sssp
    neighbors = get_sssp_neighbors(graph=graph, sssp=sssp)

    # graphics
    label = "Checking Neighbors of SSSP Nodes " + str(sssp)
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, label=label)

    # while sssp doesn't include all nodes
    while len(sssp) < num_nodes:        
        # get minimum distance node
        sssp.append(minimum_distance_neighbor_node)

        # graphics
        label = "Node " + str(minimum_distance_neighbor_node) + " With Distance " + str(minimum_distance) + " Added to SSSP"
        update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, label=label)

        # get neighbors of sssp
        neighbors = get_sssp_neighbors(graph=graph, sssp=sssp)

        # get next minimum distance edge
        minimum_distance = np.inf

        for sssp_node in sssp:
            for neighbor_node in graph.neighbors(sssp_node):
                neighbor_distance = distances[sssp_node] + graph.get_edge_data(sssp_node, neighbor_node)['weight']
                if neighbor_node not in sssp and neighbor_distance <= distances[neighbor_node]:
                    distances[neighbor_node] = neighbor_distance

                    # local memory used to help graphics
                    if neighbor_distance <= minimum_distance:
                        minimum_distance = neighbor_distance
                        minimum_distance_sssp_node = sssp_node
                        minimum_distance_neighbor_node = neighbor_node

        # graphics
        label = "Checking Neighbors of SSSP Nodes " + str(sssp)
        update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, label=label)
        
        edges.append((minimum_distance_sssp_node, minimum_distance_neighbor_node))
        
    print("Valid SSSP From ", starting_node)
    print("SSSP:", sssp)
    print("SSSP Edges:", edges)

    # graphics
    label = "SSSP Found With Edges " + str(edges)
    update_sssp_algorithm_graphics(sssp_list=sssp_list, neighbors_list=neighbors_list, distances_list=distances_list, edges_list=edges_list, labels_list=labels_list, sssp=sssp, neighbors=neighbors, distances=distances, edges=edges, label=label)

    return "Dijkstra's Algorithm: ", sssp_list, neighbors_list, distances_list, edges_list, labels_list

# GENERATE UNWEIGHTED GRAPH
def generate_unweighted_graph(nodes:list[int], edges:list[tuple]):
    """ generates unweighted graph with given nodes and edges

        * nodes:list[int] - list of nodes to add to graph
        * edges:list[tuple] - list of edges to add to graph

    """
    print("Calling Generate Graph")

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

# GENERATE WEIGHTED GRAPH
def generate_weighted_graph(nodes:list[int], edges:list[tuple]):
    """ generates weighted graph with given nodes and edges

        * nodes:list[int] - list of nodes to add to graph
        * edges:list[tuple] - list of edges to add to graph

    """
    print("Calling Generate Graph")

    graph = nx.Graph()
    print("Number of Nodes:", len(nodes))
    print("Number of Edges:", len(edges))

    # add nodes [1, number_nodes] to graph
    graph.add_nodes_from(nodes_for_adding=nodes)
    print("Number of Nodes Added:", graph.number_of_nodes())
    print("Nodes in Graph:", graph.nodes())

    # add edges [1, number_edges] to graph
    graph.add_weighted_edges_from(ebunch_to_add=edges)
    print("Number of Edges Added:", graph.number_of_edges())
    print("Edges in Graph:", graph.edges())

    return graph

# GENERATE GRAPH SEARCH ANIMATION
def generate_graph_search_animation(function:callable):
    """ generates graph, finds path from starting_node to destination_node, and animates process

        * function:callable - search algorithm to animate

    """
    print("Calling Generate Graph Search Animation")

    # generate nodes, random edges, and graph
    num_nodes = 8
    starting_node = 0
    destination_node = num_nodes - 1

    nodes = [i for i in range(0, num_nodes)]
    print(len(nodes), "Nodes Generated")

    edges = [(0, 3), (0, 1), (1, 2), (1, 4), (4, 5), (4, 6), (6, 7)]
    print(len(edges), "Edges Generated")

    graph = generate_unweighted_graph(nodes=nodes, edges=edges)

    # run search algorithm
    title, traversed_list, visited_list, labels_list = function(graph=graph, starting_node=starting_node, destination_node=destination_node)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # font 
    font = {'fontname':"Trebuchet MS"}

    # update function used to iterate through animation
    def update(frame:int):
        ax.clear()

        path_nodes = traversed_list[frame]
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)] if len(path_nodes) > 1 else []
        visited_nodes = visited_list[frame]
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # visited nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=visited_nodes, node_color="grey", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=path_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black", font_size=8)
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(traversed_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# GENERATE GRAPH SSSP ANIMATION
def generate_graph_sssp_animation(function:callable):
    """ generates graph, finds sssp, and animates process

        * function:callable - search algorithm to animate

    """
    print("Calling Generate Graph SSSP Animation")

    # generate nodes, random edges, and graph
    num_nodes = 8
    starting_node = 0
    destination_node = num_nodes - 1

    nodes = [i for i in range(0, num_nodes)]
    print(len(nodes), "Nodes Generated")

    edges = [(0, 1, 8), (0, 6, 8), (1, 2, 7), (1, 4, 4), (1, 7, 2), (6, 7, 7), (6, 5, 1), (2, 3, 9), (2, 4, 14), (3, 4, 10), (4, 5, 2), (5, 7, 6)]
    print(len(edges), "Edges Generated")

    graph = generate_weighted_graph(nodes=nodes, edges=edges)

    # run sssp algorithm
    title, sssp_list, neighbors_list, distances_list, edges_list, labels_list = function(graph=graph, starting_node=starting_node, num_nodes=num_nodes)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # font
    font = {'fontname':"Trebuchet MS"}

    # update function used to iterate through animation
    def update(frame:int):
        ax.clear()

        sssp_nodes = sssp_list[frame]
        sssp_edges = edges_list[frame]
        neighbor_nodes = neighbors_list[frame]
        distances = distances_list[frame]
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # neighbor nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=neighbor_nodes, node_color="grey", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=sssp_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=sssp_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)
        nx.draw_networkx_edge_labels(G=graph, pos=pos, ax=ax, edge_labels=edge_labels)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, labels=distances, font_color="black", font_size=8)
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(sssp_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()



# call animation functions

generate_graph_search_animation(function=find_breadthfirstsearch_path)
generate_graph_search_animation(function=find_depthfirstsearch_path)

generate_graph_sssp_animation(function=find_dijkstra_path)