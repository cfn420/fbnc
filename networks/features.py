# features.py

import numpy as np

from networks.network import x_to_matrix

class Feature:
    """
    Represents a feature function used in optimization, such as strength or stationary distribution.

    Attributes:
        name (str): Name of the feature.
        target (np.ndarray): Target value for this feature.
        value_fn (callable): Function to compute feature value from a Network.
        jacobian_fn (callable): Function to compute the Jacobian (gradient).
        projected (bool): Whether this feature is enforced via projection.
    """
    def __init__(self, name, target, value_fn=None, jacobian_fn=None, projected=False):
        self.name = name
        self.target = target
        self.value_fn = value_fn
        self.jacobian_fn = jacobian_fn
        self.projected = projected

    def value(self, net):
        """
        Evaluate the feature value on a given network.

        Parameters:
            net (Network): The network instance.

        Returns:
            np.ndarray or float: Computed value of the feature.

        Raises:
            NotImplementedError: If no value function is provided.
        """
        if self.value_fn is not None:
            return self.value_fn(net)
        else:
            raise NotImplementedError(
                f"Value function for feature '{self.name}' must be manually added if needed."
            )

    def jacobian(self, net, problem):
        """
        Evaluate the feature's Jacobian (gradient) with respect to x.

        Parameters:
            net (Network): The network instance.
            problem (ProblemInstance): The problem context.

        Returns:
            np.ndarray: Gradient matrix or vector.

        Raises:
            NotImplementedError: If no jacobian function is provided.
        """
        if self.jacobian_fn is not None:
            return self.jacobian_fn(net, problem=problem)
        else:
            raise NotImplementedError(
                f"Gradient for feature '{self.name}' must be manually added if needed."
            )

    def contributes_to_loss(self):
        """
        Indicates whether the feature contributes to the objective loss.

        Returns:
            bool: True if not projected, False otherwise.
        """
        return not self.projected
    

# Common features — directly usable by the user
def s_out(target):
    """
    Constructs an s_out (out-strength) feature with pre-defined value and Jacobian.

    Parameters:
        target (np.ndarray): Target out-strength values (1D array).

    Returns:
        Feature: A Feature instance representing out-strength.
    """
    def value_fn(net):
        return np.sum(net.W_matrix, axis=1)
    
    def jacobian_fn(net, *, problem=None, **kwargs):
        jac = np.zeros((net.n, net.x.size))

        for i in range(net.n):
            out_edge_indices = net.neighborhoods_out[i]
            jac[i, out_edge_indices] = 1.0

        return jac

    return Feature("s_out", target, value_fn, jacobian_fn, projected=True)


def s_in(target):
    """
    Constructs an s_in (in-strength) feature with pre-defined value and Jacobian.

    Parameters:
        target (np.ndarray): Target in-strength values (1D array).

    Returns:
        Feature: A Feature instance representing in-strength.
    """
    def value_fn(net):
        return np.sum(net.W_matrix, axis=0)

    def jacobian_fn(net, *, problem=None, **kwargs):
        jac = np.zeros((net.n, net.x.size)) 

        for j in range(net.n):
            in_edge_indices = net.neighborhoods_in[j] 
            jac[j, in_edge_indices] = 1.0

        return jac

    return Feature("s_in", target, value_fn, jacobian_fn, projected=True)


def HOSglobal(target):
    """
    Constructs a measure of weight imbalance.

    Parameters:
        target (float): Target value for the HOSglobal feature.

    Returns:
        Feature: A Feature instance representing HOSglobal weight imbalance.
    """
    def value_fn(net):
        s_out_vals = s_out(None).value_fn(net)
        return np.sum([ 
            (net.x[i] / s_out_vals[net.edge_matrix[i, 0]])**2
            for i in range(len(net.x))
        ])
    
    def jacobian_fn(net, *, problem=None, **kwargs):
        x = net.x  # edge weights, shape: (E,)
        edge_matrix = net.edge_matrix  # shape: (E, 2), each row is [source, target]

        s_out_feature = s_out(None)
        s_out_vals = s_out_feature.value_fn(net)         # shape: (N,)
        s_out_jac = s_out_feature.jacobian_fn(net)       # shape: (N, E)

        jac = np.zeros_like(x)  # final result: ∂f/∂x_k for each edge k

        for i in range(len(x)):
            u = edge_matrix[i, 0]        # source node of edge i
            s_out_u = s_out_vals[u]      # out-strength of source node u
            x_i = x[i]

            # Direct term: comes from ∂x_i
            direct_term = 2 * x_i / s_out_u**2
            jac[i] += direct_term

            # Indirect term: from ∂s_out[u]/∂x_k via chain rule
            ds_out_u = s_out_jac[u]  # shape: (E,)
            indirect_term = -2 * x_i**2 / s_out_u**3 * ds_out_u
            jac += indirect_term  # element-wise add to entire gradient vector

        return jac

    return Feature("HOSglobal", target, value_fn, jacobian_fn, projected=False)


def stationary_dist(target):
    """
    Constructs a stationary distribution of a stochastic matrix using the linear system solution.

    Parameters:
        target (np.ndarray): Target stationary distribution.

    Returns:
        Feature: A Feature instance representing the stationary distribution.

    Raises:
        NotImplementedError: For the Jacobian (not implemented).
    """
    def value_fn(net):
        P = net.P_matrix
        N, _ = P.shape
        Z = P - np.eye(N)
        Z[:, [0]] = np.ones((N, 1))
        pi = np.linalg.inv(Z)[0, :]
        return pi
    
    def jacobian_fn(net, *, problem=None, **kwargs):
        # not implemented, raise NotImplementedError
        raise NotImplementedError(
            "Jacobian for stationary distribution is not implemented."
        )
        
    return Feature("stationary_distribution", target, value_fn, jacobian_fn, projected=False)


def kemeny_constant(target):
    """
    Constructs a Kemeny constant feature for a Markov chain.

    Parameters:
        target (float): Target value for the Kemeny constant.

    Returns:
        Feature: A Feature instance representing the Kemeny constant.
    """
    def value_fn(net):
        P = net.P_matrix
        n = net.n
        I = np.eye(n)
        pi = stationary_dist(None).value_fn(net)
        Pi = np.tile(pi, (n, 1))
        D = np.linalg.inv(I - P + Pi) - Pi
        return np.trace(D) + 1
    
    def jacobian_fn(net, *, problem=None, **kwargs):
        n = net.n
        P = net.P_matrix
        I = np.eye(n)
        pi = stationary_dist(net).value_fn(net)
        Pi = np.tile(pi, (n ,1))
        Z = np.linalg.inv( I - P + Pi ) 
        D = Z - Pi

        # Compute the jacobian of the Kemeny constant
        def KC_dir(P, Pi, Z, D, v):   
            Pder = x_to_matrix(v, net.n, net.edge_matrix, net.bUndirected)           
            Pi_der = Pi @ Pder @ D
            Dder = -Z @ (-Pder + Pi_der) @ Z
            return np.trace(Dder)
    
        return problem.C @ np.array( [ KC_dir(P, Pi, Z, D, problem.C[:,i]) for i in range(problem.C.shape[1])] )
        
    return Feature("kemeny_constant", target, value_fn, jacobian_fn, projected=False)


