import torch
import math
from torchtyping import TensorType
from matplotlib import pyplot as plt


class PoincareDisk:
    def __init__(self):
        self._data_points = None
        self._num_points = 0

    @property
    def data_points(self):
        return self._data_points
    
    @property
    def num_points(self):
        return self._num_points


    def from_tensor(self, data: TensorType["number of points"]):
        """ Loads the data points from a torch tensor.

        Args:
            data (complex tensor):
                            torch.tensor containing the data points on the Poincare Disk.
        """
        if data.dtype is not torch.complex128:
            raise TypeError("The input tensor must be complex!")
        
        if not torch.all(torch.abs(data) < 1):
            raise ValueError("Not all Points lie inside the Poincare Disk!")
        
        self._data_points = data
        self._num_points = data.shape[0]


    def sample_random(self, num_points: int, radius: int=1):
        """ Sample random points inside the Poincare Disk using recection sampling.

        Args:
            num_points (int):
                            Number of points that get sampled.
            radius (int):
                            Radius of circle inside the Points are sampled. Standard is 1.
        """
        enough_points = False

        while not enough_points:
            reals = (-radius - radius) * torch.rand(2*num_points, dtype=torch.float64) + radius
            imags = (-radius - radius) * torch.rand(2*num_points, dtype=torch.float64) + radius
            data_points = torch.complex(reals, imags)
            data_points = data_points[torch.abs(data_points) < radius]

            if data_points.shape[0] >= num_points:
                enough_points = True

        self._data_points = data_points[:num_points]
        self._num_points = num_points


    def visualize(self, size: int, circle=False):
        """ Visualize the data points.

        Args:
            size (int):
                            The size of the pyplot figure.
            circle (bool):
                            Specifies if the circle partial D gets also plotted.
        """
        plt.figure(figsize=(size, size))
        plt.scatter(self._data_points.real, self._data_points.imag, color="peru", s=5*size)
        if circle:
            x_sphere = torch.cos(torch.linspace(0, 2 * math.pi, 1000))
            y_sphere = torch.sin(torch.linspace(0, 2 * math.pi, 1000))
            plt.plot(x_sphere, y_sphere)
        plt.axis("off")
        plt.show()


    def riemann_distance(self, z_0: TensorType[1], z_1: TensorType[1]):
        """ Computes the Riemann distance between two points in the disk.

        Args:
            z_0 (complex tensor):
                            The first data point.
            z_1 (complex tensor):
                            The second data point.

        Returns:
            distance (float):
                            Distance between the points.
        """
        return torch.log((torch.norm(1 - z_0*torch.conj(z_1)) + torch.norm(z_0 - z_1)) /
                        (torch.norm(1 - z_0*torch.conj(z_1)) - torch.norm(z_0 - z_1)))


    def riemann_distance_vec(self, points_0: TensorType["Number of elements"], points_1: TensorType["Number of elements"]):
        """ Computes the Riemannian distance between multiple points in the disk.

        Args:
            points_0 (tensor):
                            The first tensor with points.
            points_1 (tensor):
                            The second tensor with points.

        Returns:
            distance (tensor):
                            Distance between the points.
        """
        num = torch.abs(1 - points_0*torch.conj(points_1)) + torch.abs(points_0 - points_1)
        denum = torch.abs(1 - points_0*torch.conj(points_1)) - torch.abs(points_0 - points_1)
        return torch.log(num / denum)
    

    def grad_riemann_distance(self, theta, x):
        """ Computes the gradient of the Riemann distance between two points in the disk (https://arxiv.org/pdf/1705.08039.pdf).

        Args:
            theta (complex tensor):
                            The first data point.
            x (complex tensor):
                            The second data point.

        Returns:
            distance (float):
                            Distance between the points.
        """
        alpha = 1 - torch.norm(theta)**2
        beta = 1 - torch.norm(x)**2
        gamma = 1 + (2 / (alpha*beta)) * torch.norm(theta - x)**2

        return (4 / (beta * torch.sqrt(gamma**2 - 1))) * (((torch.norm(x)**2 - 2*torch.dot(theta, x) + 1) / (alpha**2)) * theta - (x / alpha))
    

    def grad_riemann_distance_vec(self, thetas_0: TensorType["Number of elements"], thetas_1: TensorType["Number of elements"]):
        """ Computes the gradient of the Riemann distance between multiple points in the disk.

        Args:
            thetas_0 (complex tensor):
                            The first tensor with points.
            thetas_1 (complex tensor):
                            The second tensor with points.

        Returns:
            distance (tensor):
                            Distances between the points.
        """
        alphas = 1 - torch.norm(thetas_0.unsqueeze(1), dim=-1)
        betas = 1 - torch.norm(thetas_1.unsqueeze(0), dim=-1)
        gamma = 1 + (2 / (alphas*betas)) * torch.norm((thetas_0 - thetas_1).unsqueeze(1), dim=-1)**2

        return (4 / (betas * torch.sqrt(gamma**2 - 1))) * (((torch.norm(thetas_1.unsqueeze(1), dim=-1)**2 - 2*thetas_0*thetas_1 + 1) / (alphas**2)) * thetas_0 - (thetas_1 / thetas_0))
    

    def distance_center(self, index: int):
        """Computes the hyperbolic Riemannian Distance to the center.

        Args:
            index (int): 
                        Index of the point.
        
        Returns:
            distance (float): 
                        The computed distance.
        """
        z = self._data_points[index]
        return torch.log((1 + z.abs()) / (1 - z.abs()))
    

    def moebius(self, index_0: int, index_1: int):
        """A Möbius transformation that sends p (index_0) to the center and is distance preserving.

        Args:
            index_0 (int):
                        Specifies the point that gets distance preserving transformed.
            index_1 (int): 
                        Specifies the point that gets sent to the center.
        """
        z = self._data_points[index_0]
        p = self._data_points[index_1]

        self._data_points[index_0] = (z - p) / (1 - torch.conj(p)*z)
        self._data_points[index_1] = 0 + 0j

    
    def exp_map(self, index: int, v: TensorType[1]):
        """Computes the exponential map on the Poincaré Disk for a single data point.

        Args:
            index (int): 
                        The starting point in the disk.
            v (complex tensor):
                        The Tangent vector.
        """
        z = self._data_points[index]
        theta = torch.angle(v)
        s = 2*torch.abs(v) / (1 - torch.abs(z)**2)

        exp_i_theta = torch.exp(1j * theta)
        exp_minus_s = torch.exp(-s)

        num = z + exp_i_theta + (z - exp_i_theta) * exp_minus_s
        den = 1 + torch.conj(z) * exp_i_theta + (1 - torch.conj(z) * exp_i_theta) * exp_minus_s
        return  num / den


    def exp_map_vec(self, index_start: int, index_end: int, v: TensorType[1]):
        """Computes the exponential map on the Poincaré Disk for multiple data points.

        Args:
            index (int): 
                        The starting point in the disk.
            v (complex tensor):
                        The Tangent vector.
        """
        batch = self._data_points[index_start:index_end]
        theta = torch.angle(v)
        s = 2*torch.abs(v) / (1 - torch.abs(batch)**2)

        exp_i_theta = torch.exp(1j * theta)
        exp_minus_s = torch.exp(-s)

        num = batch + exp_i_theta + (batch - exp_i_theta) * exp_minus_s
        den = 1 + torch.conj(batch) * exp_i_theta + (1 - torch.conj(batch) * exp_i_theta) * exp_minus_s
        return num / den