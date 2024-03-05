import torch
import math

from .graph import Graph
from .poincare import PoincareDisk
from .image_creation_funcs import geodesic
from torchtyping import TensorType
from IPython.display import clear_output
from matplotlib import pyplot as plt

class Embedding:
    def __init__(self, graph: Graph, disk: PoincareDisk):
        if graph.num_nodes != disk.num_points:
            raise AttributeError("Number of nodes in the graph and number of points in the Disk must be identical!")
        
        self._graph = graph
        self._disk = disk
        self._num_points = graph._num_nodes


    def visualize(self, size_point: float, size_fig: float, show_circle: bool, show_numbers: bool, show_geodesics: bool):
        """ Visualizes the Poincare disk with the geodesics between the points.

        Args:
            size_point (float):
                            The size of the scattered data points.
            size_fig (float):
                            The size of the pyplot figure.
            show_circle (bool):
                            Specifies of partialD should be also plotted.
            show_numbers (bool):
                            Specifies of the data points are numbered.
            show_geodesics (bool):
                            Specifies if the data points are connected with geodesics.  
        """
        fig, ax = plt.subplots(figsize=(size_fig, size_fig))
        if show_circle:
            x_sphere = torch.cos(torch.linspace(0, 2 * math.pi, 1000))
            y_sphere = torch.sin(torch.linspace(0, 2 * math.pi, 1000))
            ax.plot(x_sphere, y_sphere)
        if show_geodesics:
            ts = torch.linspace(0, 1, 100)
            for u, v in self._graph._graph.iterEdges():
                geodesic_points = geodesic(self._disk.data_points[u].unsqueeze(0), self._disk._data_points[v].unsqueeze(0), ts)
                ax.plot(geodesic_points.real, geodesic_points.imag)
        ax.scatter(self._disk._data_points.real, self._disk._data_points.imag, color="black", s=size_point)
        if show_numbers:
            for i in range(self._num_points):
                ax.text(self._disk._data_points[i].real, self._disk._data_points[i].imag, i)
        plt.axis("off")
        plt.show()


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

        Args:
            data_points (complex tensor):
                        Torch.tensor containing the data points.

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
        """ Computes the gradient of the normalized average distortions using autograd.
        
        Args:
            data_points (complex tensor):
                        Torch.tensor containing the data points.

        Returns:    
            grads (complex tensor):
                        The computed gradients.
        """
        data_points = data_points.detach().clone().requires_grad_(True)
        distortion = self.normal_avg_distortion_vec(data_points)
        distortion.backward()
        return data_points.grad
    

    def compute_alpha(self):
        """ Computes alpha for the gradient of normalized average distortion.

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
        """ Computes the gradient of the normalized distortion of one point.
        
        Args:
            index (int):
                        The index of the point.

        Returns:    
            grad (complex):
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
    

    def avg_distortion_vec(self, data_points: TensorType["Number of elements"]):
        """ Computes the average distortion vectorized.

        Args:
            data_points (complex tensor):
                        Torch.tensor containing the data points.

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

        return (2 / (n * (n-1))) * torch.sum(torch.abs(distances_disk - distances_graph) / distances_graph)
    

    def autograd_avg_distortion_vec(self, data_points: TensorType["Number of elements"]):
        """ Computes the gradient of the average distortions using autograd.
        
        Args:
            data_points (complex tensor):
                        Torch.tensor containing the data points.

        Returns:    
            grads (complex tensor):
                        The computed gradients.
        """
        data_points = data_points.detach().clone().requires_grad_(True)
        distortion = self.avg_distortion_vec(data_points)
        distortion.backward()
        return data_points.grad
    
    
    def grad_avg_distortion_vec(self, index: int):
        """ Computes the gradient of the average distortion of one point.
        
        Args:
            index (int):
                        The index of the point.

        Returns:    
            grad (float):
                        The computed gradient.
        """
        n = self._num_points

        theta = self._disk._data_points[index].detach().clone()
        thetas = torch.ones(n-1, dtype=torch.complex128) * theta
        combs = torch.cat((torch.arange(index), torch.arange(index+1, n)))

        grad_distances_disk = self._disk.grad_riemann_distance_vec(thetas, self._disk._data_points[combs])
        distances_disk = self._disk.riemann_distance_vec(thetas, self._disk._data_points[combs])

        distances_graph = self._graph.distance_all()
        distances_graph = torch.cat((distances_graph[index, 0:index], distances_graph[index, index+1:]))

        return (2 / (n*(n-1))) * torch.sum(((distances_disk - distances_graph) * grad_distances_disk) / (distances_graph * torch.abs(distances_disk - distances_graph)))
    

    def line_search(self, riemannian_grad: complex, index: int, distortion_func):
        """ Computes a suitable learning rate using the Armijo condition.

        Args:
            riemannian_grad (complex):
                        Riemannian gradient of loss.
            index (int):
                        Index of element which gets mapped.
            distortion_func (function):
                        The function used to compute the distortion.
        
        Returns:
            learning_rate (float):
                        A suitable learning rate which gurantees a sufficient improvement.
        """
        learning_rate = 2
        tau = .5
        r = 1e-4

        original_thetas = self._disk._data_points.detach().clone()
        original_theta = self._disk._data_points[index].detach().clone().unsqueeze(0)
        old_distortion = distortion_func(original_thetas)

        # initial value to force while loop
        new_distortion = old_distortion.detach().clone()

        while (old_distortion - new_distortion) < r * learning_rate * torch.norm(riemannian_grad)**2:
            learning_rate *= tau
            mapped_theta = self._disk.exp_map(original_theta, -learning_rate*riemannian_grad)
            original_thetas[index] = mapped_theta
            new_distortion = distortion_func(original_thetas)
            if math.isnan(new_distortion) or math.isinf(new_distortion):
                learning_rate *= tau
                new_distortion = old_distortion.detach().clone()
                continue

            if learning_rate < 1e-300:
                return 0
 
        return learning_rate
    

    def RSGD(self, num_epochs: int, autograd: bool, distortion: str):
        # save distortions during optimization
        distortion_per_epoch = torch.zeros(num_epochs+1)
        if distortion == "avg":
            distortion_per_epoch[0] = self.avg_distortion_vec(self._disk._data_points)
        elif distortion == "normal_avg":
            distortion_per_epoch[0] = self.normal_avg_distortion_vec(self._disk._data_points)
        else:
            raise RuntimeError("Invalid distortion selected!")
        
        for epoch in range(num_epochs):
            for i in torch.randperm(self._num_points).tolist():
                # compute euclidean gradient depending on mode
                if autograd and distortion == "avg":
                    euclid_grad = self.autograd_avg_distortion_vec(self._disk._data_points)[i].unsqueeze(0)
                elif not autograd and distortion == "avg":
                    euclid_grad = self.grad_avg_distortion_vec(i).unsqueeze(0)
                elif autograd and distortion == "normal_avg":
                    euclid_grad = self.autograd_normal_avg_distortion_vec(self._disk._data_points)[i].unsqueeze(0)
                elif not autograd and distortion == "normal_avg":
                    euclid_grad = self.grad_normal_avg_distortion_vec(i).unsqueeze(0)

                # resclae it in length to riemannian gradient
                theta = self._disk._data_points[i].detach().clone().unsqueeze(0)
                riemannian_grad_theta = ((1 - torch.norm(theta)**2) / 2) * euclid_grad

                # choose a suitable learning rate using line search
                if distortion == "avg":
                    learning_rate = self.line_search(riemannian_grad_theta, i, self.avg_distortion_vec)
                elif distortion == "normal_avg":
                    learning_rate = self.line_search(riemannian_grad_theta, i, self.normal_avg_distortion_vec)

                # update formula
                self._disk._data_points[i] = self._disk.exp_map(theta, -learning_rate*riemannian_grad_theta)
            
            if distortion == "avg":
                distortion_per_epoch[epoch+1] = self.avg_distortion_vec(self._disk._data_points)
                clear_output(wait=False)
                print(f"Average distortion after {epoch+1}. epoch: {self.avg_distortion_vec(self._disk._data_points)}.")
            elif distortion == "normal_avg":
                distortion_per_epoch[epoch+1] = self.normal_avg_distortion_vec(self._disk._data_points)
                clear_output(wait=False)
                print(f"Normalized average distortion after {epoch+1}. epoch: {self.normal_avg_distortion_vec(self._disk._data_points)}.")

        return distortion_per_epoch