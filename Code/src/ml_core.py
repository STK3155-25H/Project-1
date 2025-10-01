# -----------------------------------------------------------------------------------------
# This is where all the functions and imports are
# -----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import time
import os
from numba import njit

seed = os.environ.get("SEED")

if seed is not None:
    seed = int(seed) 
    print("Seed from env:", seed)
else:
    seed = 314
    print("Seed from hard-coded value in file ml_core.py :", seed)
    print("If you want a specific seed set the SEED environment variable")
np.random.seed(seed)
# -----------------------------------------------------------------------------------------
# Runge function
def runge_function(x, noise=False):
    """
    Compute the Runge function f(x) = 1 / (1 + 25x^2).
    
    Parameters
    ----------
    x : array_like
        Input values.
    noise : bool, optional
        If True, Gaussian noise (mean=0, std=0.5) is added. Default is False.
    
    Returns
    -------
    y : ndarray
        Output values of the Runge function (with optional noise).
    """
    # np.random.seed(seed)
    y = 1 / (1 + 25 * x**2)
    if noise:
        y += np.random.normal(0, 0.3, size=len(x), )
    return y

# -----------------------------------------------------------------------------------------
# R^2 function
def R2_score(y_true, y_pred):
    """
    Compute the coefficient of determination (R^2 score).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True target values.
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    score : float
        R^2 score.
    """
    return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)

# -----------------------------------------------------------------------------------------
# Function to calculate the MSE
def MSE(y_data,y_model):
    """
    Compute the mean squared error (MSE).

    Parameters
    ----------
    y_data : ndarray of shape (n_samples,)
        True target values.
    y_model : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    mse : float
        Mean squared error.
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# -----------------------------------------------------------------------------------------
# New polynomial features function that allows for a creation of a matrix without the intercept column
def polynomial_features(x, p, intercept=True):
    """
    Generate polynomial features up to degree p.

    Parameters
    ----------
    x : array_like of shape (n_samples,)
        Input feature vector.
    p : int
        Polynomial degree.
    intercept : bool, optional
        If True, includes a column of ones as intercept. Default is True.

    Returns
    -------
    X : ndarray of shape (n_samples, p+1) if intercept else (n_samples, p)
        Polynomial feature matrix.
    """
    x = np.array(x).reshape(-1)
    n = len(x)
    if intercept==False:
        X = np.zeros((n, p))
        for i in range(p):
            X[:, i] = x**(i+1)
    else:
        X = np.zeros((n, p+1))
        for i in range(p+1):
            X[:, i] = x**(i)
    return X

# -----------------------------------------------------------------------------------------
# New polynomial features function that allows for a creation of a matrix without the intercept column. It also scales the feature matrix
def polynomial_features_scaled(x, degree, intercept=True, col_means=None, col_stds=None, return_stats=False):
    """
    Generate a polynomial feature matrix and automatically scale each column.
    
    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Input 1D feature vector.
    degree : int
        Maximum degree of polynomial.
    intercept : bool
        Whether to include a column of ones as the intercept term.
    col_means : array-like, optional
        Means of training polynomial columns for scaling test set. Ignored if None.
    col_stds : array-like, optional
        Standard deviations of training polynomial columns for scaling test set. Ignored if None.
    return_stats : bool, optional
        If True, return the column means and stds for reuse on test set.
    
    Returns
    -------
    X_poly_scaled : ndarray of shape (n_samples, degree+1 or degree)
        Scaled polynomial feature matrix.
    col_means, col_stds : ndarray, ndarray (only if return_stats=True)
        Means and stds of polynomial columns (for consistent test scaling).
    """
    x = np.array(x).reshape(-1)
    n = len(x)
    
    # Build raw polynomial features
    if intercept:
        X = np.zeros((n, degree + 1))
        X[:, 0] = 1
        for i in range(1, degree + 1):
            X[:, i] = x ** i
    else:
        X = np.zeros((n, degree))
        for i in range(degree):
            X[:, i] = x ** (i + 1)
    
    # Scale columns (except intercept) using training stats if provided
    if intercept:
        start_col = 1
    else:
        start_col = 0
    
    if col_means is None or col_stds is None:
        # Compute stats from current data (usually train set)
        col_means = X[:, start_col:].mean(axis=0)
        col_stds = X[:, start_col:].std(axis=0)
        col_stds[col_stds == 0] = 1  # safeguard
    
    # Apply scaling
    X[:, start_col:] = (X[:, start_col:] - col_means) / col_stds
    
    if return_stats:
        return X, col_means, col_stds
    else:
        return X

# -----------------------------------------------------------------------------------------
# Splitting and scaling function for the data
def split_scale(x, y, random_state=None):
    """
    Split dataset into train/test sets and scale features.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input features.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    X_train_scaled : ndarray of shape (n_train, 1)
        Scaled training features.
    X_test_scaled : ndarray of shape (n_test, 1)
        Scaled test features.
    y_train_centered : ndarray of shape (n_train,)
        Centered training targets.
    y_test_centered : ndarray of shape (n_test,)
        Centered test targets.
    """
    rs = seed if random_state is None else random_state

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rs)
    # reshape the x datas to make them a 2d matrix
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)
    # calculate mean and std on the training set for X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1 # safe guard
    # scale the x data sets
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    # calculate mean of y
    y_mean = y_train.mean()
    # center the y data sets
    y_train_centered = y_train - y_mean
    y_test_centered = y_test - y_mean
    
    return X_train_scaled, X_test_scaled, y_train_centered, y_test_centered

# -----------------------------------------------------------------------------------------
# Gradient OLS
def gradient_OLS(X, y, theta):
    """
    Compute the gradient of the OLS cost function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Current parameter vector.

    Returns
    -------
    grad : ndarray of shape (n_features,)
        Gradient vector.
    """
    n = X.shape[0]
    grad = (-2/n) * X.T @ (y - X @ theta)
    return grad

# -----------------------------------------------------------------------------------------
# Gradient Ridge
def gradient_Ridge(X, y, theta, lam):
    """
    Compute the gradient of the Ridge cost function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Current parameter vector.
    lam : float
        Regularization strength.

    Returns
    -------
    grad : ndarray of shape (n_features,)
        Gradient vector.
    """
    n = X.shape[0]
    grad = (-2/n) * X.T @ (y - X @ theta) + 2 * lam * theta
    return grad

# -----------------------------------------------------------------------------------------
# Gradient LASSO (using subgradient descent)
def gradient_Lasso(X, y, theta, lam):
    """
    Compute the subgradient of the LASSO cost function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Current parameter vector.
    lam : float
        Regularization strength.

    Returns
    -------
    grad : ndarray of shape (n_features,)
        Subgradient vector.
    """
    n = X.shape[0]
    grad_mse = (-2/n) * X.T @ (y - X @ theta)
    grad_l1 = lam * np.sign(theta)  # subgradient of L1
    return grad_mse + grad_l1

# -----------------------------------------------------------------------------------------
# Function for the OLS
def OLS_parameters(X, y):
    """
    Compute OLS parameters using the normal equation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    theta : ndarray of shape (n_features,)
        Estimated parameters.
    """
    return (np.linalg.pinv(X.T @ X) @ X.T) @ y

# -----------------------------------------------------------------------------------------
# Ridge function
def Ridge_parameters(X, y, lam=0.01, intercept=True):
    """
    Compute Ridge regression parameters using closed-form solution.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    lam : float, optional
        Regularization strength. Default is 0.01.
    intercept : bool, optional
        If True, the intercept term is not penalized. Default is True.

    Returns
    -------
    theta : ndarray of shape (n_features,)
        Estimated parameters.
    """
    n_features = X.shape[1]
    
    if intercept:
        # Regularization matrix with 0 for intercept
        I = np.eye(n_features)
        I[0, 0] = 0  # Do not regularize the intercept
    else:
        I = np.eye(n_features)

    theta = np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y
    return theta

# -----------------------------------------------------------------------------------------
# Gradient descent code for advanced methods
def Gradient_descent_advanced(X, y, Type=0, lam=0.01, lr=0.01, n_iter=1000, tol=1e-6, method='vanilla', beta=0.9, epsilon=1e-8, batch_size=1, use_sgd=False, theta_history=False):
    """
    Gradient Descent for OLS (Type=0), Ridge (Type=1) or LASSO (Type=2)
    With advanced optimizers: Momentum, AdaGrad, RMSProp, Adam.
    With option to use stochastic gradient descent

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    Type : int, optional
        0 = OLS, 1 = Ridge, 2 = LASSO. Default is 0.
    lam : float, optional
        Regularization strength (only for Ridge). Default is 0.01.
    lr : float, optional
        Base learning rate. Default is 0.01.
    n_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    method : str, optional
        Optimization method. Options are 'vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam'. Default is 'vanilla'.
    beta : float, optional
        Momentum or decay parameter (used in momentum, RMSProp, Adam). Default is 0.9.
    epsilon : float, optional
        Small number to avoid division by zero in adaptive methods. Default is 1e-8.
    batch_size : int, optional
        The size of the batches used for sgd
    use_sgd : bool, optional
        If True, use stochastic gradient descent with random mini-batches.
        If False, use full-batch gradient descent. Default is False.
    theta_history : bool, optional
        If False, return only the final parameter vector.
        If True, return the full history of parameters over all iterations.
        Default is False.

    Returns
    -------
    theta : ndarray of shape (n_features,)
        Estimated parameters after gradient descent (if `theta_history=False`).
    history : ndarray of shape (n_iter, n_features)
        Full trajectory of parameters during training (if `theta_history=True`).
    """
    n, p = X.shape
    theta = np.zeros(p)
    history = []
    
    # Initialize variables for optimizers
    v = np.zeros(p)      # momentum
    G = np.zeros(p)      # AdaGrad
    S = np.zeros(p)      # RMSProp
    m = np.zeros(p)      # Adam first moment
    v_adam = np.zeros(p) # Adam second moment
    t = 0
    
    for epoch in range(n_iter):
        theta_old = theta.copy()

        # Choose batch
        if use_sgd:
            # Pick random mini-batch
            indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
        else:
            # Full dataset
            X_batch = X
            y_batch = y
            
        # Gradient
        if Type == 0: # OLS
            grad = gradient_OLS(X_batch, y_batch, theta)
            
        elif Type == 1: # Ridge
            grad = gradient_Ridge(X_batch, y_batch, theta, lam)
            
        elif Type == 2: # LASSO
            grad = gradient_Lasso(X_batch, y_batch, theta, lam)
            
        else:
            print("Type not found")
            break
            
        t += 1
        # Calculate theta
        if method == 'vanilla':
            theta -= lr * grad
            
        elif method == 'momentum':
            v = beta * v + (1 - beta) * grad
            theta -= lr * v
            
        elif method == 'adagrad':
            G += grad**2
            theta -= lr * grad / (np.sqrt(G) + epsilon)
            
        elif method == 'rmsprop':
            S = beta * S + (1 - beta) * grad**2
            theta -= lr * grad / (np.sqrt(S) + epsilon)
            
        elif method == 'adam':
            m = beta * m + (1 - beta) * grad
            v_adam = beta * v_adam + (1 - beta) * (grad**2)
            m_hat = m / (1 - beta**t)
            v_hat = v_adam / (1 - beta**t)
            theta -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
        else:
            print("Method not found")
            break
            
        # Add the theta value to the history list only if requested
        if theta_history:
            history.append(theta.copy())
            
        # Convergence check
        if np.linalg.norm(theta - theta_old) < tol:
            # print(f"Converged after {epoch+1} iterations, with {method}")
            break
    
    if theta_history == False:
        return theta
    else:
        return np.array(history)


def MSE_Bias_Variance(targets, predictions):
    """
    Compute the Mean Squared Error (MSE), bias^2, and variance of a set of predictions 
    compared to the true target values using the bias-variance decomposition.

    Parameters
    ----------
    target : ndarray of shape (n_samples,) or (n_bootstraps, n_samples)
        True target values. If multiple bootstrap targets are provided, only the first row is used for bias calculation.
    prediction : ndarray of shape (n_bootstraps, n_samples)
        Predicted values from multiple bootstrap models. Each row corresponds to predictions from one bootstrap.
        
    Returns
    -------
    mse : float
        Mean squared error between predictions and target values.
    bias2 : float
        Squared bias term, i.e., squared difference between the true target and the mean prediction across bootstraps.
    variance : float
        Variance term, i.e., variance of predictions across bootstraps.
    """
    mse = np.mean((targets - predictions)**2)
    bias2 = np.mean((targets - np.mean(predictions, axis=0))**2)
    variance = np.mean((predictions - np.mean(predictions, axis=0))**2)

    return mse, bias2, variance

def save_vector_with_degree(path, vec, value_name="value", degree_name="degree", std=None):
    """
    Saves a 1D vector:
      - degree, value                       (if std=None)
      - degree, value, value_std            (if std is passed)
    """
    vec = np.asarray(vec).reshape(-1)
    deg = np.arange(1, vec.shape[0] + 1)

    if std is None:
        arr = np.column_stack([deg, vec])
        header = f"{degree_name},{value_name}"
    else:
        std = np.asarray(std).reshape(-1)
        if std.shape != vec.shape:
            raise ValueError("std must have the same shape as vec")
        arr = np.column_stack([deg, vec, std])
        header = f"{degree_name},{value_name},{value_name}_std"

    np.savetxt(path, arr, delimiter=",", header=header, comments='')

def save_matrix_with_degree_cols_plus_std(path, mean_data, std_data, col_names, degree_name="degree", std_suffix="_std"):
     """
     Salva le colonne mean mantenendo i nomi originali e APPENDE, nello stesso CSV,
     le corrispondenti colonne di std come <colname><std_suffix>. La prima colonna resta 'degree'.
     """
     mean_data = np.asarray(mean_data)
     std_data = np.asarray(std_data)
     if mean_data.shape != std_data.shape:
         raise ValueError("mean_data e std_data devono avere la stessa shape")
     if mean_data.ndim != 2:
         raise ValueError("data deve essere 2D (rows=degree, cols=serie)")
     if mean_data.shape[1] != len(col_names):
         raise ValueError("len(col_names) deve coincidere con data.shape[1]")
     deg = np.arange(1, mean_data.shape[0] + 1).reshape(-1, 1)
     blocks = [mean_data] + [std_data]
     arr = np.hstack([deg] + blocks)
     header = ",".join([degree_name] + list(col_names) + [f"{c}{std_suffix}" for c in col_names])
     np.savetxt(path, arr, delimiter=",", header=header, comments='')
