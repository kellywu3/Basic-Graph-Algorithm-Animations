# Basic Graph Algorithm Animations

## OVERVIEW

This Python program generates animations for Breadth First Search and Depth First Search. The program is able to generate a random graph with a specified number of nodes, run Breadth First Search and Depth First Search on the random graph, and animate the algorithm.

**Breadth First Search**

![Alt Text](BreadthFirstSearch.gif)

**Depth First Search**

![Alt Text](DepthFirstSearch.gif)

## LIBRARIES/REQUIREMENTS

- [Matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
- [Matplotlib.animation](https://matplotlib.org/stable/api/animation_api.html)
- [NumPy](https://numpy.org)
- [NetworkX](https://networkx.org)

## TOOLS

[**pipenv**](https://pipenv.pypa.io/en/latest/)
- **Python Environment Packaging Tool**: creates flexible environment manipulation and containment by storing various environments in a common directory.
- **Dependency Resolution**: various projects may need different tool versions. dependencies versions downloaded system wide are hard to change. pipenv creates separate environments to easily switch between environments.  
- **Benefits**: creating a modifiable environment between many different projects.

start virtual environment:  
```shell
pipenv shell
```
install dependencies:
```shell
pipenv install -r requirements.txt
```
freeze requirements:
```shell
pipenv requirements > requirements.txt
```
show packages:
```shell
pipenv graph
```

[**venv**](https://docs.python.org/3/library/venv.html)
- **Python Virtual Environment Tool**: creates isolated environments by creating a subdirectory with a custom, modifiable bin directory.
- **Virtual Environment**: contains project libraries and binaries (dependencies) to isolate project software from operating system installed libraries.
- **Benefits**: creating many different switchable and testable environments for one project.

start virtual environment:
```shell
python -m venv path/to/virtual/environment
```
install dependencies:
```shell
python -m pip install -r requirements.txt
```
activate virtual environment:
```shell
source path/to/virtual/environment/bin/activate
```
deactivate virtual environment:
```shell
deactivate
```
freeze requirements:
```shell
pip freeze > requirements.txt
```

## CONCEPTS

**Breadth First Search**
- graph traversal algorithm to search for specific nodes from a starting node
- uses a queue to search graphs breadth first

pseudocode: 
```pseudo
breadth_first_search(graph, starting_node, destination node, num_nodes) {
    let path be a list
    add starting_node to path

    let queue be a list
    add path to queue

    let visited be a list
    mark starting_node as visited

    while queue not empty {
        pop queue and set path to the first entry in queue
        let current_node be the last node in path

        for neighbors neighbor_node of current_node in graph {

            if neighbor_node not visited {
                mark neighbor_node as visited
                let new_path be a copy of path
                add neighbor_node to new_path
                add new_path to queue

                if current_node is destination node {
                    return path
                }

            }

        }

    }

    return path not found

}
```

**Depth First Search**
- graph traversal algorithm to search for specific nodes from a starting node
- uses a stack to search graphs depth first

pseudocode: 
```pseudo
breadth_first_search(graph, starting_node, destination node, num_nodes) {
    let path be a list
    add starting_node to path

    let stack be a list
    add path to stack

    let visited be a list
    mark starting_node as visited

    while stack not empty {
        set path to the last entry in stack
        let current_node be the last node in path

        set neighbor_node to neighboring, unvisited node of current_node

        if neighbor_node exists {

            mark neighbor_node as visited
            let new_path be a copy of path
            add neighbor_node to new_path
            add new_path to queue

            if current_node is destination node {
                return path
            }

        } else {
            pop stack
        }

    }

    return path not found

}
```

## SOURCES

- [Matplotlib animations](https://matplotlib.org/stable/users/explain/animations/animations.html)