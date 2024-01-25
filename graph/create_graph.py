import copy
import networkit as nk
import torch
from torch.distributions import normal

def create_random_graph(
    n: int, min_b: int, max_b: int, mean: float, dev: float, progress=False
):
    """Generates a random graph, where the number of children at each node is sampled from a
       Gaussian distribution.

    Args:
        n (int): 
                    Number of nodes for the final graph.
        min_b (int): 
                    Minimum nuber of children (branching factor).
        max_b (int):
                    Maximum number of children (branching factor).
        mean (float):
                    Mean of the Gaussian distribution.
        dev (float):
                    Standard deviation of teh Gaussian distribution.
        progress (bool):
                    Boolean that specifies of all steps in the creation process
                    should be saved and also returned.
    
    Returns:
        G (nk.Graph): 
                    The created Graph.
        G_progress - optional (nk.Graph):
                    All steps during the graph creation.
    """
    distr = normal.Normal(torch.tensor([mean]), torch.tensor([dev]))

    # create graph with only the root node
    current_num_nodes = 1
    current_node = 0
    G = nk.Graph(n)

    # save progress for animation
    if progress:
        G_progress = []
        G_progress.append(copy.deepcopy(G))

    num_children = round(distr.sample().item())
    while num_children < min_b or num_children > max_b:
        num_children = round(distr.sample().item())

    # add nodes according to distribution
    while current_num_nodes + num_children <= n:
        for i in range(num_children):
            G.addEdge(current_node, current_num_nodes)
            current_num_nodes += 1

            if progress:
                G_progress.append(copy.deepcopy(G))

        current_node += 1
        num_children = round(distr.sample().item())
        while num_children < 1 or num_children > max_b:
            num_children = round(distr.sample().item())

    if current_num_nodes == n:
        if progress:
            return G, G_progress

        return G

    for i in range(n - current_num_nodes):
        G.addEdge(current_node, current_num_nodes)
        current_num_nodes += 1

        if progress:
            G_progress.append(copy.deepcopy(G))

    if progress:
        return G, G_progress

    return G