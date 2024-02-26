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


    @staticmethod
    def riemann_distance(theta_0: TensorType[1], theta_1: TensorType[1]):
        """ Computes the Riemann distance between two points in the disk.

        Args:
            theta_0 (complex tensor):
                            The first data point.
            theta_1 (complex tensor):
                            The second data point.

        Returns:
            distance (float):
                            Distance between the points.
        """
        return torch.log((torch.norm(1 - theta_0*torch.conj(theta_1)) + torch.norm(theta_0 - theta_1)) /
                        (torch.norm(1 - theta_0*torch.conj(theta_1)) - torch.norm(theta_0 - theta_1)))


    @staticmethod
    def riemann_distance_vec(thetas_0: TensorType["Number of elements"], thetas_1: TensorType["Number of elements"]):
        """ Computes the Riemannian distance between multiple points in the disk.

        Args:
            thetas_0 (tensor):
                            The first tensor with points.
            thetas_1 (tensor):
                            The second tensor with points.

        Returns:
            distance (tensor):
                            Distance between the points.
        """
        num = torch.abs(1 - thetas_0*torch.conj(thetas_1)) + torch.abs(thetas_0 - thetas_1)
        denum = torch.abs(1 - thetas_0*torch.conj(thetas_1)) - torch.abs(thetas_0 - thetas_1)
        return torch.log(num / denum)
    

    @staticmethod
    def grad_riemann_distance(theta_0: TensorType[1], theta_1: TensorType[1]):
        """ Computes the gradient of the Riemann distance between two points in the disk (https://arxiv.org/pdf/1705.08039.pdf).

        Args:
            theta_0 (complex tensor):
                            The first data point.
            theta_1 (complex tensor):
                            The second data point.

        Returns:
            distance (float):
                            Distance between the points.
        """
        alpha = 1 - torch.norm(theta_0)**2
        beta = 1 - torch.norm(theta_1)**2
        gamma = 1 + (2 / (alpha*beta)) * torch.norm(theta_0 - theta_1)**2

        return (4 / (beta * torch.sqrt(gamma**2 - 1))) * (((torch.norm(theta_1)**2 - 2*torch.dot(theta_0, theta_1) + 1) / (alpha**2)) * theta_0 - (theta_1 / alpha))
    

    @staticmethod
    def grad_riemann_distance_vec(thetas_0: TensorType["Number of elements"], thetas_1: TensorType["Number of elements"]):
        """ Computes the gradient of the Riemann distance between multiple points in the disk (https://arxiv.org/pdf/1705.08039.pdf).

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
    

    @staticmethod
    def distance_center(theta: TensorType[1]):
        """Computes the hyperbolic Riemannian Distance to the center.

        Args:
            theta_0 (complex tensor): 
                        The point.
        
        Returns:
            distance (float): 
                        The computed distance.
        """
        return torch.log((1 + theta.abs()) / (1 - theta.abs()))
    

    @staticmethod
    def moebius(theta_0: TensorType[1], theta_1: TensorType[1]):
        """A Möbius transformation that sends p (index_0) to the center and is distance preserving.

        Args:
            theta_0 (complex tensor):
                        The point that gets distance preserving transformed.
            theta_1 (complex tensor): 
                        The point that gets sent to the center.

        Returns:
            transformed_theta (complex tensor):
                        The transformed data point.
        """
        return (theta_0 - theta_1) / (1 - torch.conj(theta_1)*theta_0)
    
    
    @staticmethod
    def exp_map(theta: TensorType[1], v: TensorType[1]):
        """Computes the exponential map on the Poincaré Disk for a single data point (https://arxiv.org/pdf/1804.01882.pdf).

        Args:
            theta (complex tensor): 
                        The starting point in the disk.
            v (complex tensor):
                        The Tangent vector.

        Returns:
            new_theta (complex tensor):
                        The data point after the mapping.
        """
        lambda_theta = 2 / (1 - torch.norm(theta)**2)
        cosh_lambda_v = torch.cosh(lambda_theta * torch.norm(v))
        sinh_lambda_v = torch.sinh(lambda_theta * torch.norm(v))
        dot_theta_v = torch.dot(theta, v / torch.norm(v))

        num_0 = lambda_theta * (cosh_lambda_v + dot_theta_v * sinh_lambda_v)
        denum_0 = 1 + (lambda_theta - 1) * cosh_lambda_v + lambda_theta * dot_theta_v * sinh_lambda_v
        num_1 = (1 / torch.norm(v)) * sinh_lambda_v
        denum_1 = 1 + (lambda_theta - 1) * cosh_lambda_v + lambda_theta * dot_theta_v * sinh_lambda_v

        return (num_0 / denum_0) * theta + (num_1 / denum_1) * v


    @staticmethod
    def exp_map_vec(thetas: TensorType["Number of elements"], vs: TensorType["Number of elements"]):
        """Computes the exponential map on the Poincaré Disk for multiple data points (https://arxiv.org/pdf/1804.01882.pdf).

        Args:
            thetas (complex tensor): 
                        The starting points in the disk.
            vs (complex tensor):
                        The Tangent vectors.

        Returns:
            new_thetas (complex tensor):
                        The data points after the mapping.
        """
        lambdas_theta = 2 / (1 - torch.norm(thetas.unsqueeze(1), dim=-1)**2)
        cosh_lambdas_v = torch.cosh(lambdas_theta * torch.norm(vs.unsqueeze(1), dim=-1))
        sinh_lambdas_v = torch.sinh(lambdas_theta * torch.norm(vs.unsqueeze(1), dim=-1))
        dot_thetas_v = thetas * (vs / torch.norm(vs.unsqueeze(1), dim=-1))

        nums_0 = lambdas_theta * (cosh_lambdas_v + dot_thetas_v * sinh_lambdas_v)
        denums_0 = 1 + (lambdas_theta - 1) * cosh_lambdas_v + lambdas_theta * dot_thetas_v * sinh_lambdas_v
        nums_1 = (1 / torch.norm(vs.unsqueeze(1), dim=-1)) * sinh_lambdas_v
        denums_1 = 1 + (lambdas_theta - 1) * cosh_lambdas_v + lambdas_theta * dot_thetas_v * sinh_lambdas_v

        return (nums_0 / denums_0) * thetas + (nums_1 / denums_1) * vs
