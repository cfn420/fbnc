# sfd_algorithm.py

import logging
import numpy as np
import keyboard
from scipy import optimize
from copy import deepcopy

from networks.network import Network
from optimization.projections import projection
from optimization.feasibility import feasible
from utils import norm_Lp

logger = logging.getLogger(__name__)

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
        grad += 2 * np.sum(diff) * feature.jacobian(net, problem)

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
    gradient = loss_gradient(net, problem)
    if gradient @ delta >= 0:
        logger.warning("Gradient and descent direction are not aligned. Check the implementation.")
        raise Warning("Gradient and descent direction are not aligned. Check the implementation.")
    
    k = 0
    net_perturb = deepcopy(net)
    net_perturb.x = net.x + beta**k * alpha * delta
    while problem.loss(net) - problem.loss(net_perturb) <= -sigma * beta**k * alpha * (gradient @ delta):
        k+=1
        net_perturb.x = net.x + beta**k * alpha * delta
        if k > 100:
            # This is a safeguard against infinite loops. In practice, this should not happen.
            print(gradient @ delta)
            logger.warning("Line search exceeded 100 iterations. Check the implementation.")
            raise Warning("Line search exceeded 100 iterations. Check the implementation.")
    return k

def L2_descent(net, x_bar):
    """
    Computes the L2-normalized direction from the current state to x_bar.

    Parameters:
        net (Network): Current network state.
        x_bar (np.ndarray): Target point.

    Returns:
        np.ndarray: Normalized descent direction.
    """
    return (x_bar - net.x) / norm_Lp(x_bar - net.x, 2)

def L1_descent(net, x_bar):
    """
    Computes an L1-style descent direction (sparse direction aligned with largest gradient).

    Parameters:
        net (Network): Current network state.
        x_bar (np.ndarray): Target point.

    Returns:
        np.ndarray: Sparse descent direction vector.
    """
    delta = L2_descent(net, x_bar)
    ij = np.argmax(np.abs(delta))
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

def descent_direction(net, problem, config):
    """
    Computes a feasible descent direction based on the p-norm (L1 or L2) and network type.

    Parameters:
        net (Network): Current network.
        problem (ProblemInstance): Problem definition with norm type and feasibility info.
        config (Namespace): Configuration options.

    Returns:
        np.ndarray: Feasible descent direction.

    Raises:
        ValueError: If the norm type is invalid.
        RuntimeError: If descent direction is not aligned with the gradient.
    """
    # raise Warning("Check if all gradient computations are correct.")
    gradient = loss_gradient(net, problem)
    if problem.bMarkovian == False:
        if problem.p_norm == 1:
            delta = L1_descent(net, gradient)
        elif problem.p_norm == 2:
            delta = L2_descent(net, gradient)
        else:
            raise ValueError("Invalid p-norm specified. Only 1 and 2 are supported.")
    else:
        if problem.p_norm == 1:
            delta = L1_descent_markovian(net, problem, gradient)
        elif problem.p_norm == 2:
            delta = L2_descent_markovian(net, problem, gradient)
        else:
            raise ValueError("Invalid p-norm specified. Only 1 and 2 are supported.")

    if gradient @ delta > 0:
        logger.warning("Gradient and descent direction are not aligned. Check the implementation.")
        raise Warning("Gradient and descent direction are not aligned. Check the implementation.")

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
        # feasible(net, problem)
        vObj[k] = problem.loss(net)

        # Check convergence
        if k > 0:
            if vObj[k] < config.omega:  
                logger.info(f"Target reached at iteration {k}.")
                break
            elif abs(vObj[k] - vObj[k - 1]) < config.omega:
                logger.info(f"Converged (Δobj < {config.omega}) at iteration {k}.")
                break
            else:
                logger.info(f"Iteration {k} - Loss: {vObj[k]:.6f}")

        # Descent update
        delta = descent_direction(net, problem, config=config)
        alpha = max_stepsize(net, problem, delta, config=config)
        m = armijo_line_search(net, problem, delta, alpha, config.beta, config.sigma)
        vX[k+1] = vX[k] + config.beta**m * alpha * delta
        net.x = vX[k+1]

        # Exit if ESC pressed
        if keyboard.is_pressed('esc'):
            logger.warning("Optimization manually interrupted at iteration %d.", k)
            break
    
    return Network(mA=problem.mA, x=vX[k], bUndirected=problem.bUndirected)
