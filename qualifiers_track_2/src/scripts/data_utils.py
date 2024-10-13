import numpy as np
from nilearn.connectome import ConnectivityMeasure


def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='correlation', standardize=False).fit_transform(timeseries)
        conn[conn == 1] = 0.999999

        for i in conn:
            np.fill_diagonal(i, 0)

        conn = np.arctanh(conn)

    else:
        raise NotImplementedError

    return conn

def extract_upper_triangular(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))