"""A small collection of loss functions"""

import numpy as np
import cupy as cp

from shamboflow import IS_CUDA

def MSE(pred_val : np.ndarray, obs_val : np.ndarray) -> float :
    """Mean Squared Error

    It is a loss/cost function to
    calculate error between the
    predicted data and the actual
    data.

    `MSE = (1 / 2)(Sum (i = 1 to n)(y_i - y^_i) ^ 2)`

    It is the mean of all errors squared.
    It is multiplied with 1/2 instead of
    1/n to simplify the partial derivative.

    Args
    ----
        pred_val : ndarray
            The input vector to calculate the error for
        obs_val : ndarray
            The actual observed values
    
    Returns
    -------
        Overall error of the output

    """

    if IS_CUDA :
        obs_val = cp.asarray(obs_val)
        pred_val = cp.asarray(pred_val)
        sq_err = cp.power(cp.subtract(obs_val, pred_val), 2)
        mse_val = (1 / sq_err.size) * cp.sum(sq_err)

        return mse_val.item()

    sq_err = np.pow(np.subtract(obs_val, pred_val), 2)
    mse_val = (1 / sq_err.size) * np.sum(sq_err)

    return mse_val.item()