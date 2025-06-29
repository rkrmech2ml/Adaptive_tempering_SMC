import numpy as np
from ModelDef import model_linreg

def compute_potentials(X, particles, y_observed, sigma=0.5):
    """
    particles: list of 1D arrays (each is a candidate θ)
    X: shape (n_samples, n_features)
    y_observed: target values (shape: n_samples,)
    sigma: noise std deviation in the regression model
    """
    potentials = []
    print("the actual X values:", X)
    for p in particles:
        y_pred = model_linreg(X, p)
        # print(f"Loop {(p)}: the predicted y values :", y_pred)
        # print("observed y values:", y_observed)
        l2_error = np.sum((y_pred - y_observed) ** 2) 
        potentials.append(l2_error)
    print("the potentials:", potentials)
    return np.array(potentials)
