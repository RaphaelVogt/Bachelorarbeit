import torch
import math

from .graph import Graph
from .poincare import PoincareDisk
from torchtyping import TensorType
from IPython.display import clear_output

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
                            Index of the second point.

        Returns:
            distortion (float):
                            Pairwise distortion of the points.
        """
        return self._disk.riemann_distance(self._disk._data_points[index_0], self._disk._data_points[index_1]) / self._graph.distance_pair(index_0, index_1)
    

    def normal_avg_distortion(self):
        """ Computes the normalized average distortion.

        Returns:
            avg_distortion (float):
                        Normalized average distortion.
        """
        n = self._num_points
        num_combs = int(0.5 * n * (n-1))

        combs = torch.combinations(torch.arange(n))
        pair_ratios = torch.zeros(num_combs)

        for i, (index_0, index_1) in enumerate(combs):
            pair_ratios[i] = self.pairwise_distortion(index_0, index_1)

        pair_ratios /= torch.min(pair_ratios)
        return (2 / (n * (n-1))) * torch.sum(pair_ratios)
    

    def normal_avg_distortion_vec(self, data_points: TensorType["Number of elements"]):
        """ Computes the normalized average distortion vectorized.

        Returns:
            avg_distortion (float):
                        Normalized average distortion.
        """
        n = self._num_points

        # compute distances in the disk for all possible combinations
        combs = torch.combinations(torch.arange(n))
        distances_disk = self._disk.riemann_distance_vec(data_points[combs[:, 0]], data_points[combs[:, 1]])

        # filter the distances in the graph s.t. no duplicates are inside
        distances_graph = self._graph.distance_all()
        distances_graph = torch.flatten(torch.triu(distances_graph))
        distances_graph = distances_graph[distances_graph.nonzero()].reshape(combs.shape[0])

        # compute avg distortion on tensors
        pairwise_distortions = distances_disk / distances_graph

        return (2 / (n * (n-1))) * torch.sum(pairwise_distortions) / torch.min(pairwise_distortions)
    

    def autograd_normal_avg_distortion_vec(self, data_points: TensorType["Number of elements"]):
        distortion = self.normal_avg_distortion_vec(data_points)
        distortion.backward()
        return data_points.grad
    

    def compute_alpha(self):
        """ Computes the normalized average distortion vectorized.

        Returns:
            avg_distortion (float):
                        Normalized average distortion.
        """
        n = self._num_points

        # compute distances in the disk for all possible combinations
        combs = torch.combinations(torch.arange(n))
        distances_disk = self._disk.riemann_distance_vec(self._disk._data_points[combs[:, 0]], self._disk._data_points[combs[:, 1]])

        # filter the distances in the graph s.t. no duplicates are inside
        distances_graph = self._graph.distance_all()
        distances_graph = torch.flatten(torch.triu(distances_graph))
        distances_graph = distances_graph[distances_graph.nonzero()].reshape(combs.shape[0])

        # compute avg distortion on tensors
        pairwise_distortions = distances_disk / distances_graph
        return torch.min(pairwise_distortions)
    

    def grad_normal_avg_distortion_vec(self, index: int):
        """ Computes the gradient of the distortion of one point.
        
        Args:
            index (int):
                        The index of the point.

        Returns:    
            grad (float):
                        The computed gradient.
        """
        n = self._num_points
        alpha = self.compute_alpha()

        theta = self._disk._data_points[index].detach().clone()
        thetas = torch.ones(n-1) * theta
        combs = torch.cat((torch.arange(index), torch.arange(index+1, n)))

        grad_distances_disk = self._disk.grad_riemann_distance_vec(thetas, self._disk._data_points[combs])

        distances_graph = self._graph.distance_all()
        distances_graph = torch.cat((distances_graph[index, 0:index], distances_graph[index, index+1:]))

        return (2 / (n*(n-1))) * torch.sum(grad_distances_disk / (distances_graph * alpha))


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
        learning_rate = 2
        tau = 0.5
        r = 1e-5

        original_thetas = self._disk._data_points.detach().clone()
        original_theta = self._disk._data_points[index].detach().clone().unsqueeze(0)
        old_distortion = self.normal_avg_distortion_vec(original_thetas)

        # initial value to force while loop
        new_distortion = old_distortion.detach().clone()

        while (old_distortion - new_distortion) < r * learning_rate * torch.norm(riemannian_grad)**2:
            learning_rate *= tau
            mapped_theta = self._disk.exp_map(original_theta, -learning_rate*riemannian_grad)
            original_thetas[index] = mapped_theta
            new_distortion = self.normal_avg_distortion_vec(original_thetas)
            if math.isnan(new_distortion):
                learning_rate *= tau
                new_distortion = old_distortion.detach().clone()
                continue

        return learning_rate


    def RSGD(self, num_epochs: int):
        distortion_per_epoch = torch.zeros(num_epochs+1)
        distortion_per_epoch[0] = self.normal_avg_distortion_vec(self._disk._data_points)
        for epoch in range(num_epochs):
            for i in torch.randperm(self._num_points).tolist():
                #euclid_grads = self.autograd_normal_avg_distortion_vec(thetas)
                euclid_grad = self.grad_normal_avg_distortion_vec(i)
                theta = self._disk._data_points[i].detach().clone().unsqueeze(0)
                riemannian_grad_theta = ((1 - torch.norm(theta)**2) / 2) * euclid_grad.unsqueeze(0)
                learning_rate = self.line_search(riemannian_grad_theta, i)
                self._disk._data_points[i] = self._disk.exp_map(theta, -learning_rate*riemannian_grad_theta)

            clear_output(wait=False)
            print(f"Distortion after {epoch+1}. epoch: {self.normal_avg_distortion_vec(self._disk._data_points)}.")

            distortion_per_epoch[epoch+1] = self.normal_avg_distortion_vec(self._disk._data_points)
        return distortion_per_epoch