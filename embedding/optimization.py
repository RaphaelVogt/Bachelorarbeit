import torch
import networkit as nk
from torchtyping import TensorType
from .poincare import riemann_distance


def distortion(index_0: int, index_1: int, data_disk: TensorType, data_graph: nk.Graph):
    """Computes the distortion between two points.

    Args:
        z_0 (int): 
                    Index of the first point.
        z_1 (int): 
                    Index of the second point.
        data_disk (complex tensor):
                    Tensor with all data points in the Poincare disk.
        data_graph (nk.Graph):
                    Graph which should be embedded.
    
    Returns:
        distortion (float): 
                    The computed distortion.
    """
    distance_disk = riemann_distance(data_disk, index_0, index_1)
    dijkstra = nk.distance.BidirectionalDijkstra(data_graph, index_0, index_1)
    distance_graph = dijkstra.run().getDistance()
    return torch.abs(distance_disk - distance_graph) / distance_graph


def avg_distortion(data_disk: TensorType, data_graph: nk.Graph):
    """Computes the average distortion over all pairs of points.

    Args:
        data_disk (complex tensor):
                    Tensor with all data points in the Poincare disk.
        data_graph (nk.Graph):
                    Graph which should be embedded.
    
    Returns:
        avg_distortion (float): 
                    The computed average distortion.
    """
    num_points = data_disk.shape[0]
    combs = torch.combinations(torch.arange(num_points))
    sum_distortion = 0
    for z0, z1 in combs:
        sum_distortion += distortion(z0, z1, data_disk, data_graph)
    return sum_distortion / (0.5 * (num_points - 1) * num_points)


def create_uniform_data(num_points: int, min: float, max: float):
    """Creates a tensor with uniform sampled complex numbers.

    Args:
        num_points (int):
                    Number of points that get created.
        min (float):
                    Minimum in the sampling.
        max (float):
                    Maximum in the sampling.
    
    Returns:
        data (complex tensor): 
                    Tensor that contains the sampled numbers.
    """
    real = torch.FloatTensor(num_points, 1).uniform_(min, max)
    imag = torch.FloatTensor(num_points, 1).uniform_(min, max)
    return torch.complex(real, imag)