# utils.py

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

def norm_Lp(x, p):
    """
    Computes the Lp norm of a vector x.
    
    Parameters:
        x (np.ndarray): Input vector.
        p (int): The order of the norm (1 for L1, 2 for L2, etc.).
    
    Returns:
        float: The Lp norm of the vector x.
    """
    return np.linalg.norm(x, ord=p)

def row_normalize(M):
    """
    Normalizes the rows of a matrix so that each row sums to 1.

    Parameters:
        M (np.ndarray): Input matrix.

    Returns:
        np.ndarray: A matrix with each row normalized to sum to 1.
    """
    # Compute the sum of each row; keep dimensions for broadcasting
    row_sums = np.sum(M, axis=1, keepdims=True)
    # Avoid division by zero: if a row sum is zero, set it to one (row remains unchanged)
    row_sums[row_sums == 0] = 1
    return M / row_sums

def create_output_dir(base_dir="results"):
    """
    Creates a timestamped output directory under the specified base directory.

    Parameters:
        base_dir (str): The base directory where the output folder will be created. Defaults to "results".

    Returns:
        Path: Path to the newly created directory.
    """
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{time_stamp}")
    os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir)

def load_excel_matrix(path):
    """
    Loads data from an Excel file and returns it as a NumPy matrix.

    Parameters:
        path (str or Path): Path to the Excel file.

    Returns:
        np.ndarray: Matrix of values from the Excel file with NaNs replaced by 0.
    """
    df = pd.read_excel(path, header=None).fillna(0)
    return df.to_numpy()

def load_csv_matrix(path):
    """
    Loads data from a CSV file (semicolon-delimited) and returns it as a NumPy matrix.

    Parameters:
        path (str or Path): Path to the CSV file.

    Returns:
        np.ndarray: Matrix of values from the CSV file with NaNs replaced by 0.
    """
    df = pd.read_csv(path, sep=';', header=None).fillna(0)
    return df.to_numpy()
