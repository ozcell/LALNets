import numpy as np

def robustness_constant(acc1, acc2):
    return 1-np.abs(acc1-acc2)/acc1
