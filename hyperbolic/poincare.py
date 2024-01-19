import torch
from torchtyping import TensorType


def distance(z1: TensorType[1], z2: TensorType[1], model="disk"):
    """Computes the Riemann distance between two points in the Poincare Disk

    Args:
        z1 (complex tensor): 
                    Specifies the first point.
        z2 (complex tensor): 
                    Specifies the second point.
    
    Returns:
        distance (float): 
                    The computed distance.
    """
    # ensure that numbers are complex
    if z1.dtype is not torch.complex64 or z2.dtype is not torch.complex64:
        raise TypeError("Both numbers must be complex!")
    
    if model == "disk":
        return torch.log((torch.norm(1 - z1@torch.conj(z2)) + torch.norm(z1 - z2)) /
                        (torch.norm(1 - z1@torch.conj(z2)) - torch.norm(z1 - z2)))
    
    elif model == "plane":
        raise NotImplementedError()

    else:
        raise NotImplementedError("The model must be the Poincare Disk ('disk') or Lobachevksy half plane('plane')")


def geodesic(z1: TensorType[1], z2: TensorType[1], t, model="disk"):
    """Computes the geodesic, which is the infimum over all curves between the points

    Args:
        z1 (complex tensor): 
                    Specifies the first point.
        z2 (complex tensor): 
                    Specifies the second point.
    
    Returns:
        geodesic (complex tensor): 
                    The complex values describing the geodesic.
    """
    # ensure that numbers are complex
    if z1.dtype is not torch.complex64 or z2.dtype is not torch.complex64:
        raise TypeError("Both numbers must be complex!")
    
    # introduce the operations on the poincare disk
    circ_plus = lambda a, z : (a + z) / (1 + torch.conj(a)@z)
    circ_cross = lambda v, t : torch.tanh(t*torch.atanh(torch.norm(v))) * (v / torch.norm(v))

    if model == "disk":
        return circ_plus(z1, circ_cross(circ_plus(-z1, z2), t))
    
    elif model == "plane":
        raise NotImplementedError()

    else:
        raise NotImplementedError("The model must be the Poincare Disk ('disk') or Lobachevksy half plane('plane')")
