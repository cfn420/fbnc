#!/usr/bin/env python3
"""
main.py

This file initializes the optimization problem, constructs the network,
and runs the SPSA solver to minimize the objective.
"""

import logging
import time
from pathlib import Path
import numpy as np
from types import SimpleNamespace

from optimization.sfd_algorithm import fit
from optimization.feasibility import feasible
from networks.initialization import sample_uniform_hypercube
from networks.network import Network
from networks.features import *
from problem_instance import ProblemInstance
from utils import load_csv_matrix, create_output_dir


# ─── CONFIGURATION ───────────────────────────────────────────

# Project directories
PROJECT_ROOT    = Path(__file__).parent.resolve()
DATA_DIR        = PROJECT_ROOT / "data"
RESULTS_DIR     = PROJECT_ROOT / "results"

# FBNC problem type
FBNC_TYPE       = "what-if" # 'sampling' or 'what-if'
FEATURES        = [kemeny_constant(31.5)] # List of feature(target) to be used
NORM            = 2 # L1 norm
LB              = 0.0 # Lower bound for edge weights
UB              = 1.0 # Upper bound for edge weights

# Specific settings for sampling
N               = 10 # Number of nodes
PARAMS          = np.ones((N,N)) - np.eye(N) # Which edges may be used?
N_SAMPLES       = 2 # Number of samples to be generated

# Specific settings for what-if analysis
W_0             = load_csv_matrix(DATA_DIR / "social_network_example.csv") # Initial weights for what-if analysis
N,_             = W_0.shape
PARAMS          = np.ones((N,N)) - np.eye(N) # Which edges may be used?

# Network settings
UNDIRECTED_GRAPH  = False
MARKOVIAN_GRAPH   = True

# FBNC solver parameters
FBNC_ALG_CONFIG = SimpleNamespace(
    max_iter    = 1_000_000, # Number of iterations
    beta        = 0.5, # Armijo parameter
    sigma       = 0.5, # Armijo parameter
    alpha_ini   = 1.,  # Armijo step size parameter
    omega       = 1e-8, # Convergence threshold
)


# ─── END CONFIGURATION ────────────────────────────────────────

def run_fbnc(net, problem, logger, FBNC_ALG_CONFIG):
    """
    Runs the FBNC optimization algorithm on a given network and problem setup.

    Parameters:
        net (Network): The initial network to optimize.
        problem (ProblemInstance): The optimization problem definition.
        logger (Logger): Logger instance for progress output.
        FBNC_ALG_CONFIG (Namespace): Algorithm configuration parameters.

    Returns:
        Network: The fitted network after optimization.
    """
    logger.info("Starting FBNC…")
    t0 = time.time()
    fitted_net = fit(net, problem, config=FBNC_ALG_CONFIG)
    feasible(fitted_net, problem) # Check feasibility
    logger.info(f"Optimization finished in {time.time() - t0:.1f}s")

    return fitted_net

def main():
    """
    Entry point for running the FBNC pipeline.

    Initializes logging, sets up the problem configuration, creates output directories,
    and dispatches either sampling or what-if optimization depending on the selected mode.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    np.random.seed(12345)
    
    # Build the network graph support (that is, the adjacency matrix)
    mA = PARAMS

    # Instantiate the problem
    problem = ProblemInstance(
        fbnc_type=FBNC_TYPE,
        features=FEATURES,
        p_norm = NORM,
        mA=mA,
        lb=LB,
        ub=UB,
        bUndirected=UNDIRECTED_GRAPH,
        bMarkovian=MARKOVIAN_GRAPH,
        tol=1e-8
    )

    # Create results folder
    out_dir = create_output_dir(base_dir=str(RESULTS_DIR))

    if problem.fbnc_type == "sampling":
        problem.n_samples = N_SAMPLES

        # Sample
        for i in range(problem.n_samples):
            logger.info("Sampling {}th sample…".format(i+1))
        
            net = sample_uniform_hypercube(problem)
            fitted_net = run_fbnc(net, problem, logger, FBNC_ALG_CONFIG)

            # Save network
            fitted_net.save(out_dir / f"network_{i+1}.npz")
    
    elif problem.fbnc_type == "what-if":
        net = Network(mA=problem.mA, W=W_0, bUndirected=UNDIRECTED_GRAPH)
        fitted_net = run_fbnc(net, problem, logger, FBNC_ALG_CONFIG)
        
        # Save network
        fitted_net.save(out_dir / f"network.npz")
        
    else:
        raise ValueError("Invalid FBNC type. Choose 'sampling' or 'what-if'.")
    
if __name__ == "__main__":
    main()
