# problem_instance.py

import numpy as np
from itertools import product

from lin_alg import build_row_sum_constraints, build_col_sum_constraints, echelon_sympy, orthonormal_basis_nullspace, piv_rows
from networks.network import Network, create_edge_matrix, build_neighborhoods_out, build_neighborhoods_in, x_to_matrix
from utils import row_normalize

class ProblemInstance:
    def __init__(self,
                 fbnc_type,
                 features,
                 p_norm,
                 mA,
                 lb,
                 ub,
                 bUndirected=False,
                 bMarkovian=False,
                 tol=1e-8):
        """
        Initializes a ProblemInstance with graph structure, features, and constraints.

        Parameters:
            fbnc_type (str): Type of the FBNC model used.
            features (list): List of Feature objects defining the problem.
            p_norm (int): Order of the norm used in optimization.
            mA (np.ndarray): Adjacency matrix of the network.
            lb (float): Lower bound on edge weights.
            ub (float): Upper bound on edge weights.
            bUndirected (bool): Whether the graph is undirected. Defaults to False.
            bMarkovian (bool): Whether the network is Markovian. Defaults to False.
            tol (float): Numerical tolerance. Defaults to 1e-8.
        """
        self.fbnc_type = fbnc_type
        self.features = features
        self.p_norm = p_norm
        self.mA = mA
        self.lb = lb
        self.ub = ub
        self.bUndirected = bUndirected
        self.bMarkovian = bMarkovian
        self.tol = tol

        self.N = mA.shape[0]
        self.edge_matrix = create_edge_matrix(np.triu(self.mA) if self.bUndirected else self.mA)
        self.neighborhoods_out = build_neighborhoods_out(self.edge_matrix, self.N)
        self.neighborhoods_in = build_neighborhoods_in(self.edge_matrix, self.N)

        # Check config validity
        self.validate()

        # Projection matrices
        self.initialize_projection_matrices()

        # Save problem gradient 
        self.gradient = None 

    def loss(self, net):
        """
        Computes the sum of squared differences between actual and target values 
        for all non-projected features.

        Parameters:
            net (Network): The current network solution.

        Returns:
            float: The total loss value.
        """
        loss = 0.0

        for feature in self.features:
            if feature.projected:
                continue  # Skip features enforced by projection

            # print(f"Feature: {feature.name}, Value: {feature.value(net)}, Target: {feature.target}")

            # Support both scalar and array outputs
            diff = feature.value(net) - feature.target
            loss += np.sum(diff ** 2)

        return loss
    
    def initialize_projection_matrices(self):
        """
        Builds and stores matrices required for affine projection onto the feasible space.

        Projection logic:
            - If Markovian: enforce row-sum-to-one constraints.
            - If undirected: use one type of strength constraint if provided.
            - If directed: use both in-strength and out-strength constraints if available.
            - If no constraints are defined: use identity projection.

        Returns:
            None
        """
        A_list = []
        b_list = []

        s_in_feature = None
        s_out_feature = None

        for feature in self.features:
            name = feature.name.lower()
            if name == "s_in" and feature.projected:
                s_in_feature = feature
            elif name == "s_out" and feature.projected:
                s_out_feature = feature

        if self.bMarkovian:
            # Always impose row-sum-to-one constraints
            A_row = build_row_sum_constraints(self.mA, self.bUndirected)
            b_row = np.ones((self.N, 1))
            A_list.append(A_row)
            b_list.append(b_row)

        elif self.bUndirected:
            # Undirected: use only one type of strength constraint, if available
            chosen_feature = s_out_feature or s_in_feature
            if chosen_feature:
                mA_used = np.triu(self.mA)
                A_row = build_row_sum_constraints(mA_used, self.bUndirected)
                b_row = chosen_feature.target[:, None]
                A_list.append(A_row)
                b_list.append(b_row)

        else:
            # Directed (non-Markovian): use both if provided
            if s_out_feature:
                A_row = build_row_sum_constraints(self.mA, self.bUndirected)
                b_row = s_out_feature.target[:, None]
                A_list.append(A_row)
                b_list.append(b_row)
            if s_in_feature:
                A_col = build_col_sum_constraints(self.mA, self.bUndirected)
                b_col = s_in_feature.target[:, None]
                A_list.append(A_col)
                b_list.append(b_col)

        # If no constraints, use identity projection (C = I, A_pinv_b = 0)
        if not A_list:
            d = self.edge_matrix.shape[0]
            self.d = d
            self.C = np.eye(d)
            self.A = np.zeros((0, d))
            self.b = np.zeros(0)
            self.A_pinv_b = np.zeros(d)
            self.C__C_T_C_inv__C_T = np.eye(d)
            self.bProjectionReady = True
            return

        # Stack and reduce the constraint system
        A_comb = np.vstack(A_list)
        b_comb = np.vstack(b_list)
        A_b_comb = np.hstack([A_comb, b_comb])

        A_b_ech = echelon_sympy(A_b_comb)

        A_ech = A_b_ech[:, :-1]
        b_ech = A_b_ech[:, -1]
        pivot_rows = piv_rows(A_ech)
        A = A_ech[pivot_rows]
        b = b_ech[pivot_rows]

        self.m2, self.d = A.shape
        if np.linalg.matrix_rank(A) != len(A):
            raise Warning("Echelon operations incomplete or matrix is rank-deficient.")

        A_pinv = A.T @ np.linalg.inv(A @ A.T)
        A_pinv_b = A_pinv @ b
        C = orthonormal_basis_nullspace(A, self.m2)
        C__C_T_C_inv__C_T = C @ C.T

        # Assign projection components
        self.A = A
        self.b = b
        self.C = C
        self.A_pinv_b = A_pinv_b
        self.C__C_T_C_inv__C_T = C__C_T_C_inv__C_T
        self.bProjectionReady = True


    def validate(self):
        """
        Validates the consistency and feasibility of the problem configuration.

        Checks include:
            - Markovian assumptions on bounds and graph direction.
            - Symmetry for undirected graphs.
            - Bounds consistency.
            - NaNs in the adjacency matrix.
            - Matrix squareness.

        Raises:
            ValueError: If any inconsistency or invalid configuration is detected.
        """
        errors = []

        # Markovian graphs require weights ≤ 1
        if self.bMarkovian and self.ub > 1:
            errors.append("For Markovian graphs, upper bound on edge weights must not exceed 1.")

        if self.bMarkovian and self.lb < 0:
            errors.append("For Markovian graphs, lower bound on edge weights must not be negative.")

        if self.bMarkovian and self.bUndirected:
            errors.append("Markovian graphs cannot be undirected.")

        # Undirected graphs require symmetric adjacency matrix
        if self.bUndirected and not np.allclose(self.mA, self.mA.T):
            errors.append("Adjacency matrix must be symmetric for undirected graphs.")

        # Check for NaN values in adjacency matrix
        if np.isnan(self.mA).any():
            errors.append("Adjacency matrix contains NaN values.")

        # Bounds consistency
        if self.lb > self.ub:
            errors.append(f"Lower bound ({self.lb}) exceeds upper bound ({self.ub}).")

        # Adjacency matrix must be square
        if self.mA.shape[0] != self.mA.shape[1]:
            errors.append("Adjacency matrix must be square.")

        if errors:
            raise ValueError("Invalid ProblemInstance configuration:\n" + "\n".join(errors))
