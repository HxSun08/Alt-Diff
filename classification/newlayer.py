
import torch
import cvxpy as cp
from torch.autograd import Function
import math
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cosDis(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def decode(X_):
#     a = []
#     X = X_.cpu().numpy()
#     for i in range(len(X)):
#         a.append(X[i])
#     return a

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


def alt_diff(Pi, qi, Ai, bi, Gi, hi):
    
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device).to(torch.float64)
    sk = torch.zeros(d).to(device).to(torch.float64)
    lamb = torch.zeros(m).to(device).to(torch.float64)
    nu = torch.zeros(d).to(device).to(torch.float64)
    
    
    dxk = torch.zeros((n, n)).to(device).to(torch.float64)
    dsk = torch.zeros((d, n)).to(device).to(torch.float64)
    dlamb = torch.zeros((m, n)).to(device).to(torch.float64)
    dnu = torch.zeros((d, n)).to(device).to(torch.float64)
    
    rho = 1
    thres = 1e-3
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi.double()
    GTh = rho * Gi.T @ hi
    begin2 = time.time()

    while abs((res[-1]-res[-2])/res[-2]) > thres:
        iter_time_start = time.time()
        #print((Ai.T @ lamb).shape)
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        
        dxk = R @ (torch.eye(n).to(device) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1, n)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        res.append(0.5 * (xk.T @ Pi @ xk) + qi.T @ xk)      

    return (xk, dxk)


def diff(eps=1e-2, verbose=0):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, P_, q_, G_, h_, A_, b_):
            n, m, d = b_.shape[1], q_.shape[1], h_.shape[1]
            # print(n, m, d)
            # P = decode(P_)
            # q = q_.cpu().numpy()
            # G = G_.cpu().numpy()
            # h = h_.cpu().numpy()
            # A = A_.cpu().numpy()
            # b = b_.cpu().numpy()
            # Define and solve the CVXPY problem.
            optimal = []
            gradient = []

            for i in range(len(P_)):
                begin = time.time()
                Pi, qi, Ai, bi, Gi, hi = P_[i], q_[i], A_[i], b_[i], G_[i], h_[i]

                xk, dxk = alt_diff(Pi, qi, Ai, bi, Gi, hi)

                end = time.time()
                optimal.append(xk)
                #print('iterations:', iters)
                gradient.append(dxk)

            ctx.save_for_backward(torch.stack(gradient))
            return torch.stack(optimal)

        @staticmethod
        def backward(ctx, grad_output):
            # only call parameters q
            grad = ctx.saved_tensors

            grad_all = torch.zeros((len(grad[0]), grad[0].shape[-1])).to(device)
            for i in range(len(grad[0])):
                grad_all[i] = grad_output[i] @ grad[0][i]
            #print(grad_all.shape)
            return (None, grad_all, None, None, None, None)

    return Newlayer.apply