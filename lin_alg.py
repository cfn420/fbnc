# linear_algebra.py

import numpy as np
import sympy as sp
from itertools import product

def orthonormal_basis_nullspace(matrix, rank):
    """
    Computes an orthonormal basis for the nullspace of a matrix.

    Parameters:
        matrix (np.ndarray): Input matrix.
        rank (int): Rank of the matrix (number of independent rows).

    Returns:
        np.ndarray: Matrix whose columns form an orthonormal basis of the nullspace.
    """
    _, _, v_transpose = np.linalg.svd(matrix)
    C = np.copy(v_transpose[rank:])  # Assuming m2 is precomputed as rank(A)
    return C.T

def echelon_sympy(matrix):
    """
    Converts a matrix to row echelon form using SymPy's exact arithmetic.

    Parameters:
        matrix (array-like): Input matrix.

    Returns:
        np.ndarray: Matrix in row echelon form, with dtype float64.
    """
    sympy_matrix = sp.Matrix(matrix.tolist())
    echelon = sympy_matrix.echelon_form()
    return np.array(echelon.tolist(), dtype=np.float64)

def echelon_form(matrix):
    """
    Computes the row echelon form of a matrix using floating-point arithmetic.

    Parameters:
        matrix (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Matrix in row echelon form (a copy of the input).
    """
    B = np.copy(matrix)
    nrows, ncols = B.shape
    j = 0
    for i in range(nrows):
        pivot_row = i
        while j < ncols and np.allclose(B[pivot_row, j], 0, rtol=1e-12, atol=1e-16):
            pivot_row += 1
            if pivot_row == nrows:
                break
        if pivot_row == nrows:
            break
        if pivot_row != i:
            B[[i, pivot_row]] = B[[pivot_row, i]]
        pivot = B[i, j]
        B[i, j:] /= pivot
        for k in range(i + 1, nrows):
            factor = B[k, j] / B[i, j]
            B[k, j:] -= factor * B[i, j:]
        j += 1
    return B

def build_row_sum_constraints(mParams, bUndirected, neighborhoods_out):
    """
    Constructs the row-sum constraint matrix for weight parameters.

    Parameters:
        mParams (np.ndarray): Parameter indicator matrix of shape (N, N).
        bUndirected (bool): Whether the graph is undirected. If True, (i, j) and (j, i)
                            are treated as one parameter.

    Returns:
        np.ndarray: Constraint matrix summing outgoing weights for each node.
    """
    if bUndirected:
        
        N,N = mParams.shape
        dParam = { }
        counter = 0
        for (i,j) in product(range(N), range(N)):
            if mParams[i,j] == 1:
                dParam[ (i,j) ] = counter
                counter += 1
        
        A = np.zeros((N,int(np.sum(mParams))))
        mParams_tril = np.tril(mParams.T) # lower triangular part of matrix
        np.fill_diagonal(mParams_tril,0) # set diagonal to zero to avoid 2 values in diagonal next step.
        mA = mParams + mParams_tril
        for (i,j) in product(range(N),range(N)):
            
            if mA[i,j] == 1:
                
                if i <= j:
                    A[i, dParam.get((i,j)) ] = 1 
                else:
                    A[i, dParam.get((j,i)) ] = 1 
            
        return A
    
    else:
        N,N = mParams.shape
        param_count = int(np.sum(mParams))
        A = np.zeros((N,param_count))
        for i in range(N):
            A[i, neighborhoods_out[i]] = 1
        return A

def build_col_sum_constraints(mParams, bUndirected, neighborhoods_in):
    """
    Constructs the row-sum constraint matrix for weight parameters.

    Parameters:
        mParams (np.ndarray): Parameter indicator matrix.
        bUndirected (bool): Whether the graph is undirected.

    Returns:
        np.ndarray: Constraint matrix summing incoming weights for each node.
    """
    if bUndirected:
        raise NotImplementedError("Column sum constraints for undirected graphs are not implemented.")
    else:
        N,N = mParams.shape
        param_count = int(np.sum(mParams))
        A = np.zeros((N,param_count))
        for i in range(N):
            A[i, neighborhoods_in[i]] = 1
        return A

def piv_rows(echelon_matrix):
    """
    Identifies the indices of pivot rows in a matrix in row echelon form.

    Parameters:
        echelon_matrix (np.ndarray): Matrix in row echelon form.

    Returns:
        list of int: Indices of pivot rows (rows with leading non-zero entries).
    """
    # Identify the pivot rows and columns
    pivots = []
    pivot_columns = []
    for c, row in enumerate(echelon_matrix):
        try:
            first_nonzero_index = list(row).index(next((x for x in row if x != 0), len(row)))
        except:
            continue
            
        if first_nonzero_index not in pivot_columns:
            pivots.append(c)
            pivot_columns.append(first_nonzero_index)
            
    return pivots