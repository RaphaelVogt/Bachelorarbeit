from embedding.poincare import PoincareDisk
from embedding.graph import Graph
from embedding.embedding import Embedding

import torch
import pytest
import pytest_check as check


class TestClass:
    """
    Test creation of embedding.
    """
    def test_create(self):
        D = PoincareDisk()
        D.sample_random(100, 0.1)

        G = Graph()
        G.create_random(100, 5, 10, 7.5, 2.5)

        E = Embedding(G, D)
        check.equal(E._disk, D)
        check.equal(E._graph, G)
        check.equal(E._num_points, D._num_points)

    """
    Test vectorized normalized average distortion.
    """
    def test_normal_distortion(self):
        D = PoincareDisk()
        D.sample_random(100, 0.001)

        G = Graph()
        G.create_random(100, 5, 10, 7.5, 2.5)

        E = Embedding(G, D)

        check.equal(E.normal_avg_distortion(), pytest.approx(E.normal_avg_distortion_vec(E._disk._data_points), 1e-5))

    """
    Test gradient of vectorized normalized average distortion.
    """
    def test_grad_normal_distortion(self):
        D = PoincareDisk()
        D.sample_random(100, 0.001)

        G = Graph()
        G.create_random(100, 5, 10, 7.5, 2.5)

        E = Embedding(G, D)

        autograd_normal_dist = E.autograd_normal_avg_distortion_vec(E._disk._data_points.detach().clone().requires_grad_(True))[0]
        grad_normal_dist = E.grad_normal_avg_distortion_vec(0)
        check.is_true(torch.abs(autograd_normal_dist.real - grad_normal_dist.real) < 0.01)
        check.is_true(torch.abs(autograd_normal_dist.imag - grad_normal_dist.imag) < 0.01)