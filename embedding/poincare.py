import torch
import math
from torchtyping import TensorType
from matplotlib import pyplot as plt


class PoincareDisk:
    # get data from tensor, create random points on disk, riemann distance, geodesic, distance_center, moebius, exp_map
    def __init__(self):
        self._data_points = None


    @property
    def data_points(self):
        return self._data_points


    def from_tensor(self, data: TensorType["number of points"]):
        """ Loads the data points from a torch tensor.

        Args:
            data (complex tensor):
                            torch.tensor containing the data points on the Poincare Disk.
        """
        if data.dtype is not torch.complex64:
            raise TypeError("The input tensor must be complex!")
        
        if not torch.all(torch.abs(data) < 1):
            raise ValueError("Not all Points lie inside the Poincare Disk!")
        
        self._data_points = data


    def sample_random(self, num_points: int, radius: int=1):
        """ Sample random points inside the Poincare Disk using: https://mathworld.wolfram.com/DiskPointPicking.html.

        Args:
            num_points (int):
                            Number of points that get sampled.
            radius (int):
                            Radius of circle inside the Points are sampled. Standard is 1.
        """
        r = torch.rand(num_points) * radius
        theta = torch.rand(num_points) * 2*math.pi
        
        reals = torch.sqrt(r) * torch.cos(theta)
        imags = torch.sqrt(r) * torch.sin(theta)

        self._data_points = torch.complex(reals, imags)


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


    def riemann_distance(self, index_0: int, index_1: int):
        """ Computes the Riemann distance between two points in teh disk.

        Args:
            index_0 (int):
                            Index of teh first point.
            index_1 (int):
                            Index of the second point.

        Returns:
            distance (float):
                            Distance between the points.
        """
        z_0 = self.data_points[index_0]
        z_1 = self.data_points[index_1]

        return torch.log((torch.norm(1 - z_0*torch.conj(z_1)) + torch.norm(z_0 - z_1)) /
                        (torch.norm(1 - z_0*torch.conj(z_1)) - torch.norm(z_0 - z_1)))
    

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
        """Computes the exponential map on the Poincaré Disk.

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
        self._data_points[index] = num / den
