import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, confusion_matrix

# Custom metric
def custom_cost_only_fn(y_true, y_pred):
    """
    Return the custom cost function for the test set.   
    Range: [-1, 1], -1 
    Args:
        y_true (array-like): True labels.   
        y_pred (array-like): Predicted labels.  
    Returns:
        float: Custom cost function.
    """
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (1*tn - 10*fn) / y_true.shape[0]


