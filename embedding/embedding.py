import torch

from .graph import Graph
from .poincare import PoincareDisk

class Embedding:
    def __init__(self, graph: Graph, disk: PoincareDisk):
        self._graph = graph
        self._disk = disk
        self._num_points = graph._num_nodes


    def pairwise_distortion(self, index_0: int, index_1: int):
        """ Computes the pairwise distortion between two points.

        Args:
            index_0 (int):
                            Index of the first point.
            index_1 (int):
                            Index of teh second point.

        Returns:
            distortion (float):
                            Pairwise distortion of the points.
        """
        return self._disk.riemann_distance(index_0, index_1) / self._graph.distance_pair(index_0, index_1)
    

    def avg_distortion(self):
        """ Computes the average distortion of all points.

        Returns:
            avg_distortion (float):
                        Aevrage distortion over all points.
        """
        combs = torch.combinations(torch.arange(self._num_points))
        sum_distortion = 0
        for index_0, index_1 in combs:
            sum_distortion += self.pairwise_distortion(index_0, index_1)
        return sum_distortion / (0.5 * (self._num_points - 1) * self._num_points)
    
    def avg_distortion_vec(self):
        """ Computes the average distortion vectorized (more efficient) of all points.

        Returns:
            avg_distortion (float):
                        Aevrage distortion over all points.
        """
        combs = torch.combinations(torch.arange(self._num_points))
        distances_disk = self._disk.riemann_distance_vec(self._disk.data_points[combs[:, 0]], self._disk.data_points[combs[:, 1]])
        # TODO alle distances in graph berechnen und dann daraus die avg distortion berechnen
        ...