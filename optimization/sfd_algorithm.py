# sfd_algorithm.py

import logging
import numpy as np
from scipy import optimize
from copy import deepcopy
import sys

from networks.network import Network
from optimization.projections import projection
from optimization.feasibility import feasible
from utils import norm_Lp

logger = logging.getLogger(__name__)

def check_validity_descent_direction(gradient, delta):
    """
    Checks directional derivative of delta for invalidity by computing gradient @ delta.

    Parameters:
        gradient (np.ndarray): Gradient vector.
        delta (np.ndarray): Descent direction vector.

    Raises:
        Warning: If the gradient and descent direction are not aligned.
    """
    if gradient @ delta > 0:
        logger.warning("Gradient and descent direction are not aligned. Check the implementation.")
        logger.warning("Gradient @ delta: %.6f", gradient @ delta)
        raise Warning("Gradient and descent direction are not aligned. Check the implementation.")
    else:
        logger.info("Gradient and descent direction are aligned (negative).")

def max_stepsize(net, problem, delta, config):
    """
    Computes the maximum allowable step size in the descent direction 
    without violating box constraints.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition with bounds.
        delta (np.ndarray): Descent direction.
        config (Namespace): Configuration with initial step size.

    Returns:
        float: Maximum feasible step size.
    """
    delta = np.array(delta)
    x = np.array(net.x)

    alphas = np.full_like(delta, np.inf)

    with np.errstate(divide='ignore', invalid='ignore'):
        if problem.bMarkovian:
            # Enforce only the lower bound (problem.lb)
            mask_neg = delta < 0
            alphas[mask_neg] = (x[mask_neg] - problem.lb) / -delta[mask_neg]
        else:
            # Enforce both lower and upper bounds
            mask_pos = delta > 0
            mask_neg = delta < 0
            alphas[mask_pos] = (problem.ub - x[mask_pos]) / delta[mask_pos]
            alphas[mask_neg] = (x[mask_neg] - problem.lb) / -delta[mask_neg]

    alpha = min(config.alpha_ini, np.min(alphas))
    return alpha

def loss_gradient(net, problem):
    """
    Computes the gradient of the loss function with respect to the current network state.

    Parameters:
        net (Network): Current network state.
        problem (ProblemInstance): Problem definition containing feature targets and weights.

    Returns:
        np.ndarray: Gradient vector (same shape as net.x).
    """  
    grad = np.zeros_like(net.x)
    for feature in problem.features:
        if feature.contributes_to_loss == False:
            continue  # Skip features enforced by projection

        # Support both scalar and array outputs
        diff = feature.value(net) - feature.target
        if np.isscalar(diff) or diff.ndim == 0:
            grad += 2 * diff * feature.jacobian(net, problem=problem)
        else:
            grad += 2 * diff @ feature.jacobian(net, problem=problem)

    return grad

def armijo_line_search(net, problem, delta, alpha=1.0, beta=0.5, sigma=0.5):
    """
    Performs Armijo backtracking line search to determine a suitable step length.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition.
        delta (np.ndarray): Descent direction.
        alpha (float): Initial step size.
        beta (float): Step size reduction factor.
        sigma (float): Sufficient decrease condition.

    Returns:
        int: Number of reductions (m) applied to alpha (final step is beta^m * alpha).

    Raises:
        RuntimeError: If line search exceeds iteration limit.
    """
    k = 0
    net_perturb = deepcopy(net)
    net_perturb.x = net.x + beta**k * alpha * delta
    while problem.loss(net) - problem.loss(net_perturb) <= -sigma * beta**k * alpha * (problem.gradient @ delta):
        
        k+=1
        net_perturb.x = net.x + beta**k * alpha * delta
        feasible(net_perturb, problem)
        if k > 100:
            logger.warning(f"Line search exceeded 100 iterations. Directional derivative ({problem.gradient @ delta}) too small?.")
            sys.exit(1)

    return k

def L2_descent(net, problem, gradient):
    """
    Computes the steepest feasible descent direction for p = 2.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition.
        gradient (np.ndarray): Loss gradient.

    Returns:
        np.ndarray: Normalized steepest feasible descent direction (L2).
    """

    # Define the active set A
    A = (np.isclose(net.x, problem.lb, atol=problem.tol) & (gradient > 0)) | \
        (np.isclose(net.x, problem.ub, atol=problem.tol) & (gradient < 0))

    # Construct δ
    delta_tilde = np.zeros_like(gradient)
    delta_tilde[~A] = -gradient[~A]

    # Normalize δ
    norm = norm_Lp(delta_tilde, 2)
    if norm == 0:
        return delta_tilde  # Already optimal or stuck
    return delta_tilde / norm

def L1_descent(net, problem, gradient):
    """
    Computes an L1-style descent direction (sparse direction aligned with largest gradient).

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition.
        gradient (np.ndarray): Loss gradient.

    Returns:
        np.ndarray: Sparse descent direction vector.
    """
    delta = L2_descent(net, problem, gradient)
    abs_delta = np.abs(delta)
    max_val = np.max(abs_delta)
    max_indices = np.flatnonzero(abs_delta == max_val)
    ij = np.random.choice(max_indices)  # Randomly select among all maxima
    sign = np.sign(delta[ij])
    e = np.zeros_like(delta)
    e[ij] = sign
    return e

def L2_descent_markovian(net, problem, gradient):
    """
    Computes the L2-normalized projected gradient descent direction for Markovian networks.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition.
        gradient (np.ndarray): Loss gradient.

    Returns:
        np.ndarray: Normalized feasible descent direction.

    Raises:
        RuntimeError: If projected descent yields infeasible values.
    """    
    delta = np.zeros(problem.d)
    for i in range(net.n):
        x_i = net.x[net.neighborhoods[i]]
        gradient_i = gradient[net.neighborhoods[i]]
        I = np.array([j for j, val in enumerate(x_i) if val < problem.tol])

        def f(lamda):
            return np.sum([
                max(lamda - gradient_i[j], 0) if j in I else lamda - gradient_i[j]
                for j in range(len(x_i))
            ])

        lamda = optimize.bisect(
            f,
            a=np.min(gradient_i),
            b=np.max(gradient_i),
            xtol=problem.tol,
            rtol=problem.tol
        )

        delta_i = np.array([
            max(lamda - gradient_i[j], 0) if j in I else lamda - gradient_i[j]
            for j in range(len(x_i))
        ])
        delta[net.neighborhoods[i]] = delta_i

        # Check feasibility of updated direction
        violating_indices = [
            j for j, x_val in enumerate(x_i)
            if x_val < problem.tol and delta_i[j] < -problem.tol
        ]

        if violating_indices:
            logger.error(
                "Infeasible descent direction at node %d.\n"
                "Violating indices: %s\n"
                "Lambda: %.6f\n"
                "Neighborhood delta: %s\n"
                "Gradient: %s\n"
                "Sum of updates: %.6f",
                i,
                violating_indices,
                lamda,
                delta_i,
                gradient_i,
                np.sum(delta_i)
            )
            raise RuntimeError(f"Infeasible update at node {i}. Check logs for details.")

    delta = delta / norm_Lp(delta, 2) # normalize
    return delta

def L1_descent_markovian(net, problem, gradient):
    """
    Computes a sparse L1-style descent direction for Markovian networks.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition.
        gradient (np.ndarray): Loss gradient.

    Returns:
        np.ndarray: Sparse feasible descent direction.
    """    
    delta = L2_descent_markovian(net, problem, gradient)

    i = np.argmax( [ np.max( delta[net.neighborhoods[i]] ) - np.min( delta[net.neighborhoods[i]]) for i in range(net.n) ] ) # Find row with most potential
    
    k = np.argmax( delta[net.neighborhoods[i]] )
    j = np.argmin( delta[net.neighborhoods[i]] )
    
    k = net.neighborhoods[i][k]
    j = net.neighborhoods[i][j]
    
    delta = np.zeros(problem.d)
    delta[ k ] = 0.5
    delta[ j ] = -0.5

    return delta

def descent_direction(net, problem):
    """
    Computes a feasible descent direction based on the p-norm (L1 or L2) and network type.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition with norm type and feasibility info.

    Returns:
        np.ndarray: Feasible descent direction.

    Raises:
        ValueError: If the norm type is invalid.
        RuntimeError: If descent direction is not aligned with the gradient.
    """
    gradient = problem.gradient
    if problem.bMarkovian == False:
        if problem.p_norm == 1:
            delta = L1_descent(net, problem, gradient)
        elif problem.p_norm == 2:
            delta = L2_descent(net, problem, gradient)
        else:
            raise ValueError("Invalid p-norm specified. Only 1 and 2 are supported.")
    else:
        if problem.p_norm == 1:
            delta = L1_descent_markovian(net, problem, gradient)
        elif problem.p_norm == 2:
            delta = L2_descent_markovian(net, problem, gradient)
        else:
            raise ValueError("Invalid p-norm specified. Only 1 and 2 are supported.")

    return delta

def fit(net, problem, config=None):
    """
    Runs the FBNC optimization loop.

    Parameters:
        net (Network): Initial network state.
        problem (ProblemInstance): Problem definition.
        config (Namespace): Configuration with optimization parameters.

    Returns:
        Network: Optimized network object at convergence or early stopping.
    """
    max_iter = config.max_iter
    vX = np.zeros((max_iter, problem.d))
    vX[0] = np.copy(net.x)
    vObj = np.zeros(max_iter)

    for k in range(max_iter - 1):

        # Objective evaluation
        vObj[k] = problem.loss(net)

        # Compute descent direction
        problem.gradient = loss_gradient(net, problem)
        delta = descent_direction(net, problem)

        # Check convergence
        if k > 0:
            # check if all targets are reached
            if all(np.all(np.abs(feature.value(net) - feature.target) < config.omega) for feature in problem.features):  
                logger.info(f"Feature target(s) reached at iteration {k}.")
                break
            elif np.isclose(problem.gradient @ delta, 0):
                logger.info(f"Loss ({vObj[k]:.6f}) is not 0! But convergence reached (directional derivative ({problem.gradient @ delta:.6f}) zero) at iteration {k}.")
                break
            else:
                logger.info(f"Iteration {k} - Loss: {vObj[k]:.6f}")

        # Descent update
        alpha = max_stepsize(net, problem, delta, config=config)
        m = armijo_line_search(net, problem, delta, alpha, config.beta, config.sigma)
        vX[k+1] = vX[k] + config.beta**m * alpha * delta
        net.x = vX[k+1]
    
    return Network(mA=problem.mA, x=vX[k], bUndirected=problem.bUndirected)
