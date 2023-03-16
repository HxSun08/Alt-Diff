import numpy as np
import torch
import math

def cosDis(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def decode(X_):
    a = []
    X = X_.numpy()
    for i in range(len(X)):
        a.append(X[i])
    return a

def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss

def sgn(s):
    ss = torch.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss

def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001)) / 2
    return ss