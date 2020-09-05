import numpy as np

def compute_spectrum(X, fs):
    """
    Computes the fourier of multiple equidistantly samples signals at a time.
    
    Parameters
    ----------
    X : numpy.ndarray
        Array of signals where ``X[:,k]`` is the k-th signal.
    fs : float
        Sampling frequency.
    
    Returns
    ----------
    f : numpy.ndarray
        Freqeuency array.
    P : numpy.ndarray
        Power spectrum of the signals.
    """
    assert isinstance(X, np.ndarray)
    assert isinstance(fs, float) and fs > 0.
    assert X.ndim <= 2
    
    from scipy.fftpack import fft, fftfreq
    from scipy.signal import blackman
    n = X.shape[0]
    f = fftfreq(n, d=1. / fs)
    w = blackman(n)
    spec = fft(( (X - X.mean(axis=0) ).T * w ).T, axis=0)

    return np.abs(spec[:n//2]), f[:n//2]
