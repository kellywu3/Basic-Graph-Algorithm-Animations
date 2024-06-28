import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import random
import logging

random.seed(4)
# NODE TO INDEX
def node_to_index(node:int):
    """ converts node value to list index

        * node:int - node value

    """
    print("Calling Node to Index")

    return node - 1

# INDEX TO NODE
def index_to_node(index:int):
    """ converts list index to node value

        * index:int - index value

    """
    print("Calling Index to Node")

    return index + 1

# INITIALIZE VISITED
def initialize_visited(num_nodes:int):
    """ initializes boolean list for tracking visited nodes

        * num_nodes:int - number of nodes in graph

    """
    print("Calling Initialize Visited")

    visited = [False for i in range(num_nodes)]
    return visited

# TRANSLATE VISITED TO VISITED LIST
def translate_visited_to_visited_list(visited:list):
    """ translates boolean visited list to list of visited nodes

        * visited:list - list of boolean values indicating whether a node is visited

    """
    print("Calling Initialize Visited")

    new_list = [index_to_node(i) for i in range(len(visited)) if visited[i] == True]
    return new_list

# BREADTH FIRST SEARCH ALGORITHM
def find_breadthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using breadth first search
        returns bool, traversed_list, visited_list, labels_list indicating if the path is found and lists used for graphics

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
    queue.append(path.copy())

    # list of visited nodes
    visited = initialize_visited(num_nodes=num_nodes)
    visited[node_to_index(starting_node)] = True

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "Finding Path from Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(title)

    # while queue not empty, check each path in queue
    while len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(translate_visited_to_visited_list(visited=visited))
        title = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(title)

        # check each neighbor
        neighbors = sorted(graph.neighbors(current_node))
        for neighbor_node in neighbors:

            # for each not visited neighbor, mark as visited, add to path, and push path to queue
            if not visited[node_to_index(neighbor_node)]:
                visited[node_to_index(neighbor_node)] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path.copy())

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Neighbor Node " + str(neighbor_node) + " Visited"
                labels_list.append(title)

                # if node at end of path is destination, return path found
                if neighbor_node == destination_node:
                    print("Valid Path from ", starting_node, "to", destination_node)
                    print("Path:", new_path)

                    # graphics
                    traversed_list.append(new_path.copy())
                    visited_list.append(translate_visited_to_visited_list(visited=visited))
                    title = "Path Found at " + str(new_path)
                    labels_list.append(title)

                    return "Breadth First Search: ", True, traversed_list, visited_list, labels_list
                
                # graphics
                traversed_list.append(path)
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Checking Neighbors of Node " + str(current_node)
                labels_list.append(title)

    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)

    # graphics
    traversed_list.append(path.copy())
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "No Path Found"
    labels_list.append(title)

    return "Breadth First Search: ", False, traversed_list, visited_list, labels_list

# BREADTH FIRST SEARCH RECURSIVE HELPER
def find_breadthfirstsearch_path_recurse(graph:nx.Graph, starting_node:int, destination_node:int, queue:list[list], visited:list[bool], traversed_list:list[list], visited_list:list[list], labels_list:list[str]):
    """ recursive function used in find_breadthfirstsearch_path_recursively

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * queue:list[list] - list of paths used in as queue in search
        * paths:list[list] - list of paths returned
        * visited:list[bool] - list of boolean values indicating whether a node is visited
        * traversed_list:list[list] - list of lists of traversed nodes for graphics
        * visited_list:list[list] - list of lists of visited nodes for graphics
        * labels_list:list[str] - list of strings of titles for graphics

    """
    print("Calling Breadth First Search Recursive Helper")

    if len(queue) > 0:
        # set path to path popped from queue
        path = queue.pop(0)

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(translate_visited_to_visited_list(visited=visited))
        title = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(title)

        # check each neighbor
        neighbors = sorted(graph.neighbors(current_node))
        for neighbor_node in neighbors:

            # for each not visited neighbor, mark as visited, add to path, and push path to queue
            if not visited[node_to_index(neighbor_node)]:
                visited[node_to_index(neighbor_node)] = True
                new_path = path.copy()
                new_path.append(neighbor_node)
                queue.append(new_path.copy())

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Neighbor Node " + str(neighbor_node) + " Visited"
                labels_list.append(title)

                # if node at end of path is destination, return path found
                if neighbor_node == destination_node:
                    print("Valid Path from ", starting_node, "to", destination_node)
                    print("Path:", new_path)

                    # graphics
                    traversed_list.append(new_path.copy())
                    visited_list.append(translate_visited_to_visited_list(visited=visited))
                    title = "Path Found at " + str(new_path)
                    labels_list.append(title)

                    return "Breadth First Search: ", True, traversed_list, visited_list, labels_list
                
                # graphics
                traversed_list.append(path)
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Checking Neighbors of Node " + str(current_node)
                labels_list.append(title)

        # call recursive function
        return find_breadthfirstsearch_path_recurse(graph=graph, starting_node=starting_node, destination_node=destination_node, queue=queue, visited=visited, traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list)
    
    else:
        # if destination not found, return path not found 
        print("No Valid Path from", starting_node, "to", destination_node)
        return "Breadth First Search: ", False, traversed_list, visited_list, labels_list

# BREADTH FIRST SEARCH RECURSIVE ALGORITHM
def find_breadthfirstsearch_path_recursively(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using breadth first search recursive
        returns bool, traversed_list, visited_list, labels_list indicating if the path is found and lists used for graphics

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
    queue.append(path.copy())

    # list of visited nodes
    visited = initialize_visited(num_nodes=num_nodes)
    visited[node_to_index(starting_node)] = True

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "Finding Path from Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(title)

    # call recursive function
    return find_breadthfirstsearch_path_recurse(graph=graph, starting_node=starting_node, destination_node=destination_node, queue=queue, visited=visited, traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list)

# DEPTH FIRST SEARCH ALGORITHM
def find_depthfirstsearch_path(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using depth first search
        returns bool, traversed_list, visited_list, labels_list indicating if the path is found and lists used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * num_nodes:int - number of nodes in graph

    """
    print("Calling Depth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    stack = []
    stack.append(path.copy())

    # list of visited nodes
    visited = initialize_visited(num_nodes=num_nodes)
    visited[node_to_index(starting_node)] = True

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "Finding Path from Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(title)

    # while stack not empty, check each path in queue
    while len(stack) > 0:
        # set path to path from stack
        path = stack[-1]

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(translate_visited_to_visited_list(visited=visited))
        title = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(title)

        # check if unvisited neighbor exists
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_node = None
        for node in neighbors:
            if not visited[node_to_index(node)]:
                neighbor_node = node
                break

        # for the first not visited neighbor, mark as visited, add to path, and push path to stack
        if neighbor_node:
            visited[node_to_index(neighbor_node)] = True
            new_path = path.copy()
            new_path.append(neighbor_node)
            stack.append(new_path.copy())

            # graphics
            traversed_list.append(new_path.copy())
            visited_list.append(translate_visited_to_visited_list(visited=visited))
            title = "Neighbor Node " + str(neighbor_node) + " Visited"
            labels_list.append(title)

            # if node at end of path is destination, return path found
            if neighbor_node == destination_node:
                print("Valid Path from ", starting_node, "to", destination_node)
                print("Path:", new_path)

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Path Found at " + str(new_path)
                labels_list.append(title)

                return "Depth First Search: ", True, traversed_list, visited_list, labels_list
            
        else:
            # graphics
            path = stack.pop(-1)
            traversed_list.append(path)
            visited_list.append(translate_visited_to_visited_list(visited=visited))
            title = "No Neighbors Found"
            labels_list.append(title)

    # if destination not found, return path not found 
    print("No Valid Path from", starting_node, "to", destination_node)

    # graphics
    traversed_list.append(path.copy())
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "No Path Found"
    labels_list.append(title)

    return "Depth First Search: ", False, traversed_list, visited_list, labels_list

# DEPTH FIRST SEARCH RECURSIVE HELPER
def find_depthfirstsearch_path_recurse(graph:nx.Graph, starting_node:int, destination_node:int, stack:list[list], visited:list[bool], traversed_list:list[list], visited_list:list[list], labels_list:list[str]):
    """ recursive function used in find_depthfirstsearch_path_recursively

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * stack:list[list] - list of paths used in as stack in search
        * paths:list[list] - list of paths returned
        * visited:list[bool] - list of boolean values indicating whether a node is visited
        * traversed_list:list[list] - list of lists of traversed nodes for graphics
        * visited_list:list[list] - list of lists of visited nodes for graphics
        * labels_list:list[str] - list of strings of titles for graphics

    """
    print("Calling Breadth First Search Recursive Helper")

    if len(stack) > 0:
        # set path to path from stack
        path = stack[-1]

        # get node at end of path
        current_node = path[-1]

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(translate_visited_to_visited_list(visited=visited))
        title = "Checking Neighbors of Node " + str(current_node)
        labels_list.append(title)

        # check if unvisited neighbor exists
        neighbors = sorted(graph.neighbors(current_node))
        neighbor_node = None
        for node in neighbors:
            if not visited[node_to_index(node)]:
                neighbor_node = node
                break

        # for the first not visited neighbor, mark as visited, add to path, and push path to stack
        if neighbor_node:
            visited[node_to_index(neighbor_node)] = True
            new_path = path.copy()
            new_path.append(neighbor_node)
            stack.append(new_path.copy())

            # graphics
            traversed_list.append(new_path.copy())
            visited_list.append(translate_visited_to_visited_list(visited=visited))
            title = "Neighbor Node " + str(neighbor_node) + " Visited"
            labels_list.append(title)

            # if node at end of path is destination, return path found
            if neighbor_node == destination_node:
                print("Valid Path from ", starting_node, "to", destination_node)
                print("Path:", new_path)

                # graphics
                traversed_list.append(new_path.copy())
                visited_list.append(translate_visited_to_visited_list(visited=visited))
                title = "Path Found at " + str(new_path)
                labels_list.append(title)

                return "Depth First Search: ", True, traversed_list, visited_list, labels_list
            
        else:
            # graphics
            path = stack.pop(-1)
            traversed_list.append(path)
            visited_list.append(translate_visited_to_visited_list(visited=visited))
            title = "No Neighbors Found"
            labels_list.append(title)
        
        return find_depthfirstsearch_path_recurse(graph=graph, starting_node=starting_node, destination_node=destination_node, stack=stack, visited=visited, traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list)

    else: 
        # if destination not found, return path not found 
        print("No Valid Path from", starting_node, "to", destination_node)

        # graphics
        traversed_list.append(path.copy())
        visited_list.append(translate_visited_to_visited_list(visited=visited))
        title = "No Path Found"
        labels_list.append(title)

        return "Depth First Search: ", False, traversed_list, visited_list, labels_list

# DEPTH FIRST SEARCH RECURSIVE ALGORITHM
def find_depthfirstsearch_path_recursively(graph:nx.Graph, starting_node:int, destination_node:int, num_nodes:int):
    """ finds shortest path in the graph between the starting node and the destination node using depth first search recursively
        returns bool, traversed_list, visited_list, labels_list indicating if the path is found and lists used for graphics

        * graph:nx.Graph - graph to find path
        * starting_node:int - index of the starting node
        * destination_node:int - index of the destination node
        * num_nodes:int - number of nodes in graph

    """
    print("Calling Recursive Depth First Search")

    # final path
    path = []
    path.append(starting_node)

    # list of paths
    stack = []
    stack.append(path.copy())

    # list of visited nodes
    visited = initialize_visited(num_nodes=num_nodes)
    visited[node_to_index(starting_node)] = True

    # list of all traversed paths, visited paths, and labels for graphics
    traversed_list = []
    traversed_list.append(path.copy())
    visited_list = []
    visited_list.append(translate_visited_to_visited_list(visited=visited))
    title = "Finding Path from Node " + str(starting_node) + " to Node " + str(destination_node)
    labels_list = []
    labels_list.append(title)

    return find_depthfirstsearch_path_recurse(graph=graph, starting_node=starting_node, destination_node=destination_node, stack=stack, visited=visited, traversed_list=traversed_list, visited_list=visited_list, labels_list=labels_list)


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
    title, path_found, traversed_list, visited_list, labels_list = function(graph=graph, starting_node=starting_node, destination_node=destination_node, num_nodes=num_nodes)

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

        # start and end node frame
        nx.draw_networkx_nodes(G=graph, pos=pos, ax=ax, nodelist=(starting_node, destination_node), node_color="white", edgecolors="black", node_size=450, linewidths=2)

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

# ANIMATE BREADTH FIRST SERARCH
def animate_breadth_first_search(num_nodes:int):
    """ animates breadth first search algorithm

        * num_nodes:int - number of nodes in graph

    """
    print("Calling Generate Breadth First Search Animation")

    num_edges = random.randrange(start=0, stop=get_maximum_number_edges(num_nodes) + 1)
    starting_node = 1
    destination_node = num_nodes

    generate_graph_animation(num_nodes=num_nodes, num_edges=num_edges, starting_node=starting_node, destination_node=destination_node, function=find_breadthfirstsearch_path)

# ANIMATE DEPTH FIRST SERARCH
def animate_breadth_first_search(num_nodes:int):
    """ animates depth first search algorithm

        * num_nodes:int - number of nodes in graph

    """
    print("Calling Generate Depth First Search Animation")

    num_edges = random.randrange(start=0, stop=get_maximum_number_edges(num_nodes) + 1)
    starting_node = 1
    destination_node = num_nodes

    generate_graph_animation(num_nodes=num_nodes, num_edges=num_edges, starting_node=starting_node, destination_node=destination_node, function=find_depthfirstsearch_path)

# call animation functions
animate_breadth_first_search(num_nodes=8)
animate_breadth_first_search(num_nodes=8)