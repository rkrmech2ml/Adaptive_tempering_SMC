import numpy as np

def model_linreg(X, theta):
    """
    Linear regression model.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    theta : np.ndarray
        Parameter vector of shape (n_features,) or
        2D array of shape (n_particles, n_features)

    Returns:
    --------
    y_pred : np.ndarray
        Predicted target values.
        If theta is 1D: shape (n_samples,)
        If theta is 2D: shape (n_samples, n_particles)
    """
    if theta.ndim == 1:
        #print("the theta is 1D:", theta)
        return X @ theta
    
          # Single parameter vector
    else:
        #print("the theta is 2D:", theta)
        return X @ theta.T  # Multiple particles
