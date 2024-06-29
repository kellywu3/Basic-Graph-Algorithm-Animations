import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random
import logging

random.seed(4)

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
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(visited.copy())
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(label)

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(visited.copy())
        label = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(label)

        # check each neighbor
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_nodes = []
        for node in neighbors:
            if node not in visited:
                neighbor_nodes.append(node)

        # graphics
        if len(neighbor_nodes) == 0:
            traversed_list.append(path.copy())
            visited_list.append(visited.copy())
            label = "No Neighbors Found"
            labels_list.append(label)

        else:
            for neighbor_node in neighbor_nodes:
                # for each not visited neighbor, mark as visited, add to path, and push path to queue
                visited.append(neighbor_node)
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path.copy())

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(visited.copy())
                label = "Neighbor Node " + str(neighbor_node) + " Visited"
                labels_list.append(label)

                # if node at end of path is destination, return path found
                if neighbor_node == destination_node:
                    print("Valid Path From ", starting_node, "to", destination_node)
                    print("Path:", new_path)

                    # graphics
                    traversed_list.append(new_path.copy())
                    visited_list.append(visited.copy())
                    label = "Path Found at " + str(new_path)
                    labels_list.append(label)

                    return "Breadth First Search: ", traversed_list, visited_list, labels_list
                    
                # graphics
                traversed_list.append(path.copy())
                visited_list.append(visited.copy())
                label = "Checking Neighbors of Node " + str(current_node)
                labels_list.append(label)

                # graphics
                if neighbor_node == neighbor_nodes[len(neighbor_nodes) - 1]:
                    traversed_list.append(path.copy())
                    visited_list.append(visited.copy())
                    label = "No Neighbors Found"
                    labels_list.append(label)

    # if destination not found, return path not found 
    print("No Valid Path From", starting_node, "to", destination_node)

    # graphics
    traversed_list.append(path.copy())
    visited_list.append(visited.copy())
    label = "No Path Found"
    labels_list.append(label)

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
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(visited.copy())
    label = "Finding Path From Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(label)

    # while stack not empty, check each path in queue
    while len(stack) > 0:
        # set path to path from stack
        path = stack[-1]

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(visited.copy())
        label = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(label)

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
            traversed_list.append(new_path.copy())
            visited_list.append(visited.copy())
            label = "Neighbor Node " + str(neighbor_node) + " Visited"
            labels_list.append(label)

            # if node at end of path is destination, return path found
            if neighbor_node == destination_node:
                print("Valid Path From ", starting_node, "to", destination_node)
                print("Path:", new_path)

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(visited.copy())
                label = "Path Found at " + str(new_path)
                labels_list.append(label)

                return "Depth First Search: ", traversed_list, visited_list, labels_list
            
        else:
            # graphics
            path = stack.pop(-1)
            traversed_list.append(path.copy())
            visited_list.append(visited.copy())
            label = "No Neighbors Found"
            labels_list.append(label)

    # if destination not found, return path not found 
    print("No Valid Path From", starting_node, "to", destination_node)

    # graphics
    traversed_list.append(path.copy())
    visited_list.append(visited.copy())
    label = "No Path Found"
    labels_list.append(label)

    return "Depth First Search: ", traversed_list, visited_list, labels_list

# DIJKSTRA'S ALGORITHM
def find_dijkstra_path(graph:nx.Graph, starting_node:int, num_nodes:int):
    """ finds shortest path in the graph between all nodes using dikstra's algorithm
        returns 

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the first graph node
        * num_nodes:int - number of nodes in graph

    """
    print("Calling Dijkstra's Algorithm")

    # final sssp
    sssp = []

    # list of distances from starting node to all nodes
    distances = {i:np.inf for i in range(1, num_nodes + 1)}
    distances[starting_node] = 0

    # list of all traversed paths, visited paths, and labels for graphics
    sssp_list = []
    sssp_list.append(sssp.copy())
    neighbors_list = []
    neighbors_list.append(sssp.copy())
    label = "Finding SSSP From Node " + str(starting_node)
    labels_list = []
    labels_list.append(label)

    # while sssp doesn't include all nodes
    while len(sssp) < num_nodes:
        non_sssp_distances = {i:distances[i] for i in range(1, num_nodes + 1) if i not in sssp}
        minimum_distance_node = min(non_sssp_distances, key=non_sssp_distances.get)
        sssp.append(minimum_distance_node)

        # graphics
        sssp_list.append(sssp.copy())
        neighbors_list.append(sssp.copy())
        label = "Node " + str(starting_node) + " With Minimum Distance " + str(distances[minimum_distance_node]) + " Added to SSSP"
        labels_list.append(label)

        for node in sssp:
            for neighbor_node in graph.neighbors(node):
                distances[neighbor_node] = min(distances[neighbor_node], distances[node] + graph.get_edge_data(node, neighbor_node)['weight'])
        
    print("Valid SSSP From ", starting_node)
    print("SSSP:", sssp)

    # graphics
    sssp_list.append(sssp.copy())
    neighbors_list.append(sssp.copy())
    label = "SSSP Found at " + str(sssp)
    labels_list.append(label)

    return "Dijkstra's Algorithm: ", sssp_list, neighbors_list, labels_list

# GET MAXIMUM NUMBER EDGES
def get_maximum_number_edges(num_nodes:int):
    """ finds the maximum number of edges for graph with num_nodes nodes

    * num_nodes:int - number of nodes in graph

    """
    print("Calling Get Maximum Number Edges")

    max_num_edges = num_nodes * (num_nodes - 1) // 2
    print("Maximum Number Edges:", max_num_edges)
    return max_num_edges

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
    starting_node = 1
    destination_node = num_nodes

    nodes = [i for i in range(1, num_nodes + 1)]
    print(len(nodes), "Nodes Generated")

    edges = [(1, 4), (1, 2), (2, 3), (2, 5), (5, 6), (5, 7), (7, 8)]
    print(len(edges), "Edges Generated")

    graph = generate_unweighted_graph(nodes=nodes, edges=edges)

    # run search algorithm
    title, traversed_list, visited_list, labels_list = function(graph=graph, starting_node=starting_node, destination_node=destination_node)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

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
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black")
        
        font = {'fontname':"Trebuchet MS"}
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(traversed_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# GENERATE GRAPH SSSP ANIMATION
def generate_graph_sssp_animation(function:callable):
    """ generates graph, finds path from sssp, and animates process

        * function:callable - search algorithm to animate

    """
    print("Calling Generate Graph SSSP Animation")

    # generate nodes, random edges, and graph
    num_nodes = 8
    starting_node = 1
    destination_node = num_nodes

    nodes = [i for i in range(1, num_nodes + 1)]
    print(len(nodes), "Nodes Generated")

    edges = [(1, 2, 8), (1, 7, 8), (2, 3, 7), (2, 5, 4), (2, 8, 2), (7, 8, 7), (7, 6, 1), (3, 4, 9), (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 8, 6)]
    print(len(edges), "Edges Generated")

    graph = generate_weighted_graph(nodes=nodes, edges=edges)

    # run sssp algorithm
    title, sssp_list, neighbors_list, labels_list = function(graph=graph, starting_node=starting_node, num_nodes=num_nodes)

    pos = nx.spring_layout(G=graph)
    fig, ax = plt.subplots()

    # update function used to iterate through animation
    def update(frame:int):
        ax.clear()

        sssp_nodes = sssp_list[frame]
        sssp_edges = [(sssp_nodes[i], sssp_nodes[i + 1]) for i in range(len(sssp_nodes) - 1)] if len(sssp_nodes) > 1 else []
        neighbor_nodes = neighbors_list[frame]
        label = labels_list[frame]

        # background frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=graph.edges(), edge_color="black", width=1)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=graph.nodes(), node_color="white", edgecolors="black", node_size=400, linewidths=1)

        # visited nodes frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=neighbor_nodes, node_color="grey", edgecolors="black", node_size=450, linewidths=1)

        # animation nodes frame
        nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, edgelist=sssp_edges, edge_color="yellow", width=2)
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=sssp_nodes, node_color="yellow", edgecolors="black", node_size=450, linewidths=2)

        # labels frame
        nx.draw_networkx_labels(G=graph, pos=pos, ax=ax, font_color="black")
        
        font = {'fontname':"Trebuchet MS"}
        ax.set_title(title, **font)
        ax.set_xlabel(label, **font)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(sssp_list), interval=1200, repeat=True, repeat_delay=1200)
    plt.show()

# call animation functions
# generate_graph_search_animation(function=find_breadthfirstsearch_path)
# generate_graph_search_animation(function=find_depthfirstsearch_path)

generate_graph_sssp_animation(function=find_dijkstra_path)