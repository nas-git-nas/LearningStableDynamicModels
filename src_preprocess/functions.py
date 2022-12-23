import numpy as np

def polyFct2(x, a1, a2):
    return a1*x + a2*x*x

def polyFct3(x, a1, a2, a3):
    return a1*x + a2*(x**2) + a3*(x**3)

def polyFct4(x, a1, a2, a3, a4):
    return a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4)

def polyFct5(x, a1, a2, a3, a4, a5):
    return a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4) + a5*(x**5)

def expRise(X, tau, delay):
    """
    First order model
    Args:
        X: (time, y_final)
        tau: time const.
        delay: delay
    Returns:
        y: value at time t
    """
    (t, y_final) = X
    return np.maximum(0, y_final*(1 - np.exp(-(t - delay)/tau)))
    #return np.sign(gain)*np.maximum(0, np.abs(gain)*(1 - np.exp(-(t - delay)/tau)))

def expFall(X, tau, delay):
    """
    First order model
    Args:
        X: (time, y_init)
        tau: time const.
        delay: delay
    Returns:
        y: value at time t
    """
    (t, y_init) = X
    return np.minimum(y_init, y_init*np.exp(-(t - delay)/tau))