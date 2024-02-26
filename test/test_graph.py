from embedding.graph import Graph

import pytest
import pytest_check as check


class TestClass:
    """
    Test creation of empty graph
    """
    def test_create_empty(self):
        G = Graph()
        check.is_true(G._graph == None)
        check.is_true(G._mean == 0)
        check.is_true(G._num_nodes == 0)
        check.is_true(G._stddev == 0)

    """
    Test random graph creation
    """
    def test_create_random(self):
        G = Graph()
        G.create_random(100, 0, 10, 5.0, 2.5)
        check.is_true(G._mean == 5.0)
        check.is_true(G._stddev == 2.5)
        check.is_true(G._num_nodes == 100)
        check.is_true(G._graph.numberOfNodes() == 100)

    """
    Test if errors are risen correctly
    """
    def test_error_rise(self):
        G = Graph()
        with pytest.raises(PermissionError):
            G.distance_all()

        with pytest.raises(PermissionError):
            G.distance_pair(0, 1)

        with pytest.raises(PermissionError):
            G.create_random(100, 0, 10, 5.0, 2.5)
            G.create_random(100, 0, 10, 5.0, 2.5)