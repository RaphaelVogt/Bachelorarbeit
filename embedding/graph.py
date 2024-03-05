import networkit as nk
import torch
from torch.distributions import normal


class Graph:
    def __init__(self):
        self._num_nodes = 0
        self._mean = 0
        self._stddev = 0
        self._graph = None

    @property
    def graph(self):
        return self._graph
    
    @property
    def num_nodes(self):
        return self._num_nodes


    def load_from_file(self, filename: str):
        """Loads the data from a specified file and creates the graph.

        Args:
            filename (string):
                            path to the file where the graph is saved
        """
        raise NotImplementedError()


    def create_random(self, num_nodes: int, min_b: int, max_b: int, mean: float, dev: float):
        """Generates a random graph, where the number of children at each node is sampled from a
       Gaussian distribution.

        Args:
            num_nodes (int): 
                        Number of nodes for the final graph.
            min_b (int): 
                        Minimum nuber of children (branching factor).
            max_b (int):
                        Maximum number of children (branching factor).
            mean (float):
                        Mean of the Gaussian distribution.
            dev (float):
                        Standard deviation of the Gaussian distribution.
        """
        if self._graph is not None:
            raise PermissionError("The graph has already been created!")
        
        self._num_nodes = num_nodes
        self._mean = mean
        self._stddev = dev

        distr = normal.Normal(torch.tensor([mean]), torch.tensor([dev]))

        # create graph with only the root node
        current_num_nodes = 1
        current_node = 0
        G = nk.Graph(num_nodes)

        num_children = round(distr.sample().item())
        while num_children < min_b or num_children > max_b:
            num_children = round(distr.sample().item())

        # add nodes according to distribution
        while current_num_nodes + num_children <= num_nodes:
            for i in range(num_children):
                G.addEdge(current_node, current_num_nodes)
                current_num_nodes += 1

            current_node += 1
            num_children = round(distr.sample().item())
            while num_children < min_b or num_children > max_b:
                num_children = round(distr.sample().item())

        if current_num_nodes == num_nodes:
            self._graph = G
            return

        for i in range(num_nodes - current_num_nodes):
            G.addEdge(current_node, current_num_nodes)
            current_num_nodes += 1

        self._graph = G


    def distance_pair(self, index_0: int, index_1: int):
        """Computes the path distance between two points in the graph.

        Args:
            index_0 (int): 
                            Index of first node.
            index_1 (int):
                            Index of second node.
        Returns:
            distance (int):
                            Distance of shortest path between the nodes.
        """
        if self._graph is None:
            raise PermissionError("The graph has not yet been created!")

        dijkstra = nk.distance.BidirectionalDijkstra(self._graph, index_0, index_1)
        dijkstra.run()
        return dijkstra.getDistance()


    def distance_all(self):
        """Computes the distances between all pairs of points.

        Returns:
            distances (torch.tensor):
                            Pairwise distances between all nodes of the graph.
        """
        if self._graph is None:
            raise PermissionError("The graph has not yet been created!")
        
        apsp = nk.distance.APSP(self._graph)
        apsp.run()
        return torch.tensor(apsp.getDistances())