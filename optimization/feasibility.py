# feasibility.py

import logging
import numpy as np

logger = logging.getLogger(__name__)

def feasible(net, problem, tol=1e-8, return_W=False):
    """
    Check that a candidate decision vector x yields a valid weight matrix W.

    Parameters:
        problem (ProblemInstance): The problem setup.
        net (object): Must have attributes `x` (decision vector) and `W_matrix` (N x N weight matrix).
        tol (float): Numerical tolerance for feasibility checks.
        return_W (bool): Whether to return the W matrix if feasible.

    Returns:
        np.ndarray: The N×N weight matrix W if return_W is True and all checks pass.
        None: If return_W is False and all checks pass.

    Raises:
        ValueError: If any feasibility condition fails.
    """
    x = np.copy(net.x)
    W = np.copy(net.W_matrix)

    # 1. Check stochastic matrix row sums if Markovian
    if problem.bMarkovian:
        row_sums = np.sum(W, axis=1)
        if not np.allclose(row_sums, 1.0, atol=tol):
            logger.error("Row sums incorrect for W: %s", row_sums)
            raise ValueError(f"Row sums of W not equal to 1 (within tol={tol}).")

    # 2. Check bounds on x
    if np.any(x < (problem.lb - tol)):
        x_min = np.min(x)
        logger.error("Decision vector x below lower bound: min(x)=%.4g, lb=%.4g", x_min, problem.lb)
        raise ValueError(f"Some entries in x below lower bound (lb={problem.lb}, tol={tol}).")

    if np.any(x > (problem.ub + tol)):
        x_max = np.max(x)
        logger.error("Decision vector x above upper bound: max(x)=%.4g, ub=%.4g", x_max, problem.ub)
        raise ValueError(f"Some entries in x exceed upper bound (ub={problem.ub}, tol={tol}).")

    # 3. Check symmetry if undirected
    if problem.bUndirected:
        if not np.allclose(W, W.T, atol=tol):
            logger.error("Transition matrix W is not symmetric for undirected graph.")
            raise ValueError("W must be symmetric when bUndirected=True.")

    if return_W:
        return W
