# network.py

import numpy as np
from itertools import product
        
from utils import row_normalize

class Network:
    def __init__(self, mA, x=None, W=None, bUndirected=False):
        """
        Initializes a Network instance with a vector or matrix of weights.

        Parameters:
            mA (np.ndarray): Adjacency matrix indicating allowed edges.
            x (np.ndarray, optional): Vector of edge parameters.
            W (np.ndarray, optional): Weighted adjacency matrix to be converted into vector form.
            bUndirected (bool, optional): Whether the network is undirected. Defaults to False.

        Raises:
            ValueError: If both x and W are provided.
        """
        self.n = mA.shape[0]
        self.bUndirected = bUndirected
        self.mA = mA

        if x is not None and W is not None:
            raise ValueError("Only one of 'x' or 'W' should be provided.")

        if self.bUndirected == False:
            self.edge_matrix = create_edge_matrix(self.mA)
        else:
            self.edge_matrix = create_edge_matrix(np.triu(self.mA))

        if W is not None:
            self.x = matrix_to_x(W, self.mA, self.bUndirected)
        elif x is not None:
            self.x = x
        else:
            self.x = np.zeros(len(self.edge_matrix))

        self.neighborhoods_out = build_neighborhoods_out(self.edge_matrix, self.n)
        self.neighborhoods_in = build_neighborhoods_in(self.edge_matrix, self.n)
        
    
    # ----------------------------
    # Conversion Methods
    # ----------------------------
    @property
    def W_matrix(self):
        """Returns the weighted adjacency matrix obtained by directly mapping vector entries to edges."""
        return x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected)
        
    @property
    def P_matrix(self):
        """Returns the stochastic matrix obtained by directly mapping vector entries to edges."""
        if self.bUndirected == False:
            return x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected)
        else:
            return row_normalize(x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected))
        
    def save(self, filename):
        """
        Saves the current network state to a `.npz` file.

        Parameters:
            filename (str): Path of the file to save the network to.
        """
        np.savez(filename, x=self.x, mA=self.mA, bUndirected=self.bUndirected)


def matrix_to_x(W, mA, bUndirected):
    """
    Extracts the vector representation from a matrix W by reading the entries at positions
    where mA (the binary matrix) is one.
    
    Parameters:
        W (np.ndarray): Transition matrix.
        mA (np.ndarray): Binary matrix indicating allowed edges.
        bUndirected (bool): If True, only include each undirected edge once (i <= j).
    
    Returns:
        np.ndarray: The vector of edge weights
    """
    N, _ = mA.shape
    if not bUndirected:
        return np.array([W[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1])
    else:
        return np.array([W[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1 and i <= j])  

def x_to_matrix(x, N, edge_matrix, bUndirected):
    """
    Converts a vector of edge weights into a square matrix.

    Parameters:
        x (np.ndarray): Vector of edge values.
        N (int): Size of the resulting square matrix.
        edge_matrix (np.ndarray): List of (i, j) index pairs for placing x values.
        bUndirected (bool): Whether to symmetrize the matrix.

    Returns:
        np.ndarray: A matrix of shape (N, N) with values placed according to edge_matrix.
    """
    P = np.zeros((N, N))
    P[edge_matrix[:, 0], edge_matrix[:, 1]] = x
    if bUndirected:
        return P + P.T
    else:
        return P
    
def remove_nodes(matrix, exclude):
    """
    Returns a submatrix obtained by excluding the rows and columns given in 'exclude'.
    (Uses the implementation from functions.py.)
    
    Parameters:
        exclude (list or np.ndarray): Indices to exclude.
        matrix (np.ndarray, optional): Matrix to subset. If None, self.direct_P() is used.
    
    Returns:
        tuple: (mSubset_matrix, mask) where mSubset_matrix is the submatrix and mask is the boolean mask.
    """
    n, _ = matrix.shape
    mask = np.full((n, n), True)
    mask[exclude] = False
    mask[:, exclude] = False
    mSubset = matrix[mask]
    mSubset_matrix = mSubset.reshape(n - len(exclude), n - len(exclude))
    return mSubset_matrix, mask

def build_node_mask(n, exclude):
    """
    Removes specified nodes from a square matrix by excluding their rows and columns.

    Parameters:
        matrix (np.ndarray): Input square matrix.
        exclude (list or np.ndarray): Indices to exclude.

    Returns:
        tuple:
            - np.ndarray: Reduced square matrix with excluded nodes removed.
            - np.ndarray: Boolean mask of the retained indices.
    """
    node_mask = np.ones(n, dtype=bool)
    node_mask[exclude] = False
    return node_mask

def create_edge_matrix(mA):
    """
    Creates an edge matrix from an adjacency matrix mA.
    
    Parameters:
        mA (np.ndarray): Adjacency matrix.
    
    Returns:
        np.ndarray: An array of shape (num_edges, 2) where each row is an edge (i, j) with mA[i, j] nonzero.
    """
    indices = np.nonzero(mA)
    num_edges = indices[0].shape[0]
    E = np.zeros((num_edges, 2), dtype=int)
    E[:, 0] = indices[0]
    E[:, 1] = indices[1]
    return E

def build_neighborhoods_out(edge_matrix, N):
    """
    Builds a subset list for projection, grouping vector indices by source node for outgoing edges.

    Parameters:
        edge_matrix (np.ndarray): An array of shape (num_edges, 2), where each row is an edge (i, j).
        N (int): Number of nodes in the network.

    Returns:
        list of list[int]: neighborhoods[i] contains the indices of edges starting from node i.
    """
    neighborhoods = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edge_matrix):
        neighborhoods[i].append(idx)
    return neighborhoods

def build_neighborhoods_in(edge_matrix, N):
    """
    Builds a subset list for projection, grouping vector indices by source node for ingoing edges.

    Parameters:
        edge_matrix (np.ndarray): An array of shape (num_edges, 2), where each row is an edge (i, j).
        N (int): Number of nodes in the network.

    Returns:
        list of list[int]: neighborhoods[i] contains the indices of edges going to node i.
    """
    neighborhoods = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edge_matrix):
        neighborhoods[j].append(idx)
    return neighborhoods

