import torch

from .graph import Graph
from .poincare import PoincareDisk
from torchtyping import TensorType

class Embedding:
    def __init__(self, graph: Graph, disk: PoincareDisk):
        if graph.num_nodes != disk.num_points:
            raise AttributeError("Number of nodes in the graph and number of points in the Disk must be identical!")
        
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
        return self._disk.riemann_distance(self._disk._data_points[0], self._disk._data_points[1]) / self._graph.distance_pair(index_0, index_1)
    

    def avg_distortion(self, index_start: int, index_end: int):
        """ Computes the average distortion of a data batch.

        Args:
            index_start (int):
                        Start index of the data batch.
            index_end (int):
                        End index of the data batch.

        Returns:
            avg_distortion (float):
                        Average distortion of the batch.
        """
        n = index_end - index_start
        combs = torch.combinations(torch.arange(index_start, index_end))
        sum_distortion = 0
        for index_0, index_1 in combs:
            sum_distortion += self.pairwise_distortion(index_0, index_1)
        return sum_distortion / (0.5 * (n - 1) * n)
    

    def avg_distortion_vec(self):
        """ Computes the average distortion vectorized.

        Returns:
            avg_distortion (float):
                        Average distortion of the batch.
        """
        n = self._num_points
        # compute distances in the disk for all possible combinations
        combs = torch.combinations(torch.arange(0, n))
        distances_disk = torch.zeros(combs.shape[0])
        for i, (index_0, index_1) in enumerate(combs):
            distances_disk[i] = self._disk.riemann_distance(index_0, index_1)

        # filter the distances in the graph s.t. no duplicates are inside
        distances_graph = self._graph.distance_all()
        distances_graph = torch.flatten(torch.triu(distances_graph))
        distances_graph = distances_graph[distances_graph.nonzero()].reshape(combs.shape[0])

        # compute avg distortion on tensors
        pairwise_distortions = (distances_disk / torch.min(distances_disk)) / (distances_graph / torch.min(distances_graph))
        sum_distortion = torch.sum(pairwise_distortions)
        return ((2 / (n*(n-1)))) * sum_distortion
    

    def euclid_grad_avg_distortion(self, index: int):
        """ Computes the gradient of the distortion of one point.
        
        Args:
            index (int):
                        The index of the point.

        Returns:    
            grad (float):
                        The computed gradient.
        """
        n = self._num_points
        z = self._disk._data_points[index]
        zs = torch.ones(self._num_points)*z
        distances_graph_all = self._graph.distance_all()[0, :]
        distances_graph = torch.cat((distances_graph_all[0:index], distances_graph_all[index+1:]))
        grad_distances_disk_all = self._disk.grad_riemann_distance_vec(zs, self._disk._data_points)
        grad_distances_disk = torch.cat((grad_distances_disk_all[0:index], grad_distances_disk_all[index+1:]))
        return (1 / n) * torch.sum(distances_graph * grad_distances_disk)


    def euclid_grad_avg_distortion_vec(self, index_start: int, index_end: int):
        """ Computes the gradient of the average distortion vectorized (more efficient) of a data batch.

        Args:
            index_start (int):
                        Start index of the data batch.
            index_end (int):
                        End index of the data batch.

        Returns:
            grad_avg_distortion (float):
                        Gradient of average distortion of the batch.
        """
        n = index_end - index_start

        # filter the distances in the graph s.t. no duplicates are inside
        combs = torch.combinations(torch.arange(0, n))
        distances_graph = self._graph.distance_all()[index_start:index_end, index_start:index_end]
        distances_graph = torch.flatten(torch.triu(distances_graph))
        distances_graph = distances_graph[distances_graph.nonzero()].reshape(combs.shape[0])

        # compute gradient of distortion
        distances_graph = 1 / distances_graph
        grad_distance_disk = self._disk.grad_riemann_distance_vec(self._disk.data_points[combs[:, 0]], self._disk.data_points[combs[:, 1]])
        return (2 / (n * (n-1))) * torch.sum(distances_graph * grad_distance_disk)

        
    def line_search(self, riemannian_grad: complex, index: int):
        """ Computes a suitable learning rate using the Armijo condition.

        Args:
            riemannian_grad_batch (complex):
                        Riemannian gradient of loss over batch.
            index_start (int):
                        Start index of data batch.
            index_end (ind):
                        End index of data batch.
        
        Returns:
            eta (float):
                        A suitable learning rate which gurantees a sufficient improvement.
        """
        eta = 1
        tau = 0.5
        r = 1e-4
        n = self._num_points

        old_distortion = self.avg_distortion_vec()
        data_before_map = self._disk._data_points.detach()

        self._disk._data_points[index] = self._disk.exp_map(index, -eta*riemannian_grad)
        new_distortion = self.avg_distortion_vec()

        while (old_distortion - new_distortion) < r * eta * torch.norm(riemannian_grad)**2:
            self._disk._data_points = data_before_map.detach()
            eta = tau * eta
            old_distortion = self.avg_distortion_vec()
            data_before_map = self._disk._data_points.detach()

            self._disk._data_points[index] = self._disk.exp_map(index, -eta*riemannian_grad)
            new_distortion = self.avg_distortion_vec()



    def RSGD(self, num_epochs: int):
        for epoch in range(num_epochs):
            for i in torch.randperm(self._num_points).tolist():
                theta = self._disk._data_points[i]
                euclid_grad = self.euclid_grad_avg_distortion(i)
                riemannian_grad_batch = ((1 - torch.norm(theta)**2) / 2) * euclid_grad
                self.line_search(riemannian_grad_batch, i)

                
            print(self.avg_distortion_vec())




    def mini_batch_RSGD(self, batch_size: int, num_epochs: int, loss_func: str):
        if loss_func == "avg_dist":
            n = 0

            for i in range(num_epochs):
                mini_batch = self._disk.data_points[n:n+batch_size].detach()
                euclid_grad_batch = self.euclid_grad_avg_distortion_vec(n, n+batch_size)
                riemannian_grad_batch = ((1 - torch.norm(mini_batch)**2) / 2) * euclid_grad_batch
                learning_rate = self.line_search(riemannian_grad_batch, n, n+batch_size)
                self._disk._data_points = self._disk.exp_map_vec(0, self._num_points, -learning_rate*riemannian_grad_batch)
                n = n + batch_size

                print(self.avg_distortion_vec())

                if n + batch_size > self._num_points - 1:
                    n = 0
                    self._disk._data_points = self._disk._data_points[torch.randperm(self._num_points)]

                