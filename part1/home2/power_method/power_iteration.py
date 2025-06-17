import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data:np.ndarray, num_steps:int):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    
    b_k = np.random.rand(n)
    
    for _ in range(num_steps):
        b_k1 = data @ b_k
        
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        
    eigenvalue = float(b_k.T @ (data @ b_k))
    return float(eigenvalue), b_k