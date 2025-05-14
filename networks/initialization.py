# initialization.py

import numpy as np

from optimization.projections import projection
from optimization.feasibility import feasible
from networks.network import Network

def sample_uniform_hypercube(problem):
    """
    Samples a single feasible initial solution by projecting a random point 
    from the unit hypercube and checking feasibility.

    Parameters:
        problem (ProblemInstance): A problem instance containing the dimension (d),
                                   projection matrix, network matrix (mA), and 
                                   graph structure flag (bUndirected).

    Returns:
        Network: A Network object initialized with a feasible solution vector.
    """
    x = np.random.uniform(size=problem.d)
    x = projection(x, problem)
    net = Network(
        mA=problem.mA,
        x=x,
        bUndirected=problem.bUndirected
    )
    feasible(net, problem)  # Raises a warning if not feasible

    return net

