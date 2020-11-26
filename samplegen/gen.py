import numpy as np
import scipy.stats as ss

def random_dist(data, N):
    """Genrate random N items from data"""
    return np.random.choice(data, size=N)

def random_normal_dist(data, N, left=0.01, right=0.99):
    data_size = len(data)
    x = np.linspace(ss.norm.ppf(left), ss.norm.ppf(right), data_size)
    probs = (ss.norm.pdf(x)/ss.norm.pdf(x).sum())
    
    return np.random.choice(data, size=N, p=probs)

def left_skew_dist(data, N, left=0.01, right=0.99, a=2, loc=2):
    data_size = len(data)
    x = np.linspace(ss.gamma.ppf(left, a, loc=loc), ss.gamma.ppf(right, a, loc=loc), data_size)
    probs = (ss.gamma.pdf(x, a)/ss.gamma.pdf(x, a).sum())
    
    return np.random.choice(data, size=N, p=probs)

def right_skew_dist(data, N, left=0.01, right=0.99, a=2, loc=2):
    data_size = len(data)
    x = np.linspace(ss.gamma.ppf(left, a, loc=loc), ss.gamma.ppf(right, a, loc=loc), data_size)
    probs = (ss.gamma.pdf(x, a)/ss.gamma.pdf(x, a).sum())[::-1]
    
    return np.random.choice(data, size=N, p=probs)
