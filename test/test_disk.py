from embedding.poincare import PoincareDisk

import torch
import pytest
import pytest_check as check


class TestClass:
    """
    Test creation of empty disk
    """
    def test_create_empty(self):
        D = PoincareDisk()
        check.is_true(D._data_points == None)
        check.is_true(D._num_points == 0)

    """
    Test disk creation from tensor
    """
    def test_create_tensor(self):
        D = PoincareDisk()

        data_points_invalid = torch.complex(torch.rand(100, dtype=torch.float64)*2, torch.rand(100, dtype=torch.float64)*2)
        with pytest.raises(ValueError):
            D.from_tensor(data_points_invalid)

        data_points_invalid = torch.rand(100)
        with pytest.raises(TypeError):
            D.from_tensor(data_points_invalid)

        data_points = torch.complex(torch.rand(100, dtype=torch.float64)*0.5, torch.rand(100, dtype=torch.float64)*0.5)
        D.from_tensor(data_points)
        check.is_true(D._num_points == 100)
        for data_tensor, data_disk in zip(data_points, D._data_points):
            check.equal(data_tensor, data_disk)

    """
    Test random disk creation
    """
    def test_create_random(self):
        D = PoincareDisk()
        D.sample_random(100, 0.5)
        check.is_true(D._num_points == 100)
        check.is_true(D._data_points.dtype == torch.complex128)
        for point in D._data_points:
            check.is_true(torch.abs(point).item() <= 0.5)

    """
    Test computation of Riemannian distance
    """
    def test_distance(self):
        D = PoincareDisk()
        D.sample_random(100, 1)

        indices_0 = torch.arange(0, 100)
        indices_1 = torch.randperm(100)
        vectorized_distances = D.riemann_distance_vec(D._data_points, D._data_points[indices_1])
        for i, j in zip(indices_0, indices_1):
            distance = D.riemann_distance(D._data_points[i], D._data_points[j])
            torch.equal(distance, vectorized_distances[i])

    """
    Test computation of gradient of Riemannian distance
    """
    def test_grad_distance(self):
        ...