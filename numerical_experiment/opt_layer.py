import torch
import cvxpy as cp
from torch.autograd import Function
import math
import numpy as np
import time
from util import *
from qpth.qp import QPFunction
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpylayers.torch import CvxpyLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def alt_diff(Pi, qi, Ai, bi, Gi, hi):
    
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device)
    sk = torch.zeros(d).to(device)
    lamb = torch.zeros(m).to(device)
    nu = torch.zeros(d).to(device)
    
    
    dxk = torch.zeros((n,m)).to(device)
    dsk = torch.zeros((d,m)).to(device)
    dlamb = torch.zeros((m,m)).to(device)
    dnu = torch.zeros((d,m)).to(device)
    
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
        
        coef1 = Ai.T @ dlamb + Gi.T @ dnu - rho * Ai.T + rho * Gi.T @ dsk
        dxk = R @ coef1
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1,m)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk - torch.eye(m).to(device))

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        res.append(0.5 * (xk.T @ Pi @ xk) + qi.T @ xk)

    return (xk, dxk)

def cvxpylayer(Pi, qi, Ai, bi, Gi, hi):

    P_np = Pi.cpu().numpy()
    q_np = qi.cpu().numpy()
    A_np = Ai.cpu().numpy()
    G_np = Gi.cpu().numpy()
    h_np = hi.cpu().numpy()
    
    b0 = cp.Parameter(Ai.shape[0])
    x = cp.Variable(qi.shape[0])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, psd_wrap(P_np)) + q_np.T @ x), [A_np @ x == b0, G_np @ x <= h_np])
    
    layer = CvxpyLayer(prob, parameters=[b0], variables=[x])
    #print(layer)

    
    # solution, = layer(bi, solver_args={'mode': 'dense', 'eps': 1e-4})
    solution, = layer(bi, solver_args={'mode': 'dense'})
    solution.sum().backward()
    b_g = bi.grad.cpu().numpy()
    

    x_cvx = solution.detach()

    print("The optimal value is", (1/2) * x_cvx.T @ Pi @ x_cvx + qi.T @ x_cvx)

    return (x_cvx, b_g)

def optnet(Pi, qi, Ai, bi, Gi, hi):
    # P_np = Pi.cpu().numpy()
    # q_np = qi.cpu().numpy()
    # A_np = Ai.cpu().numpy()
    # b_np = bi.cpu().numpy()
    # G_np = Gi.cpu().numpy()
    # h_np = hi.cpu().numpy()
    nBatch = 1

    P1 = Pi.unsqueeze(0).expand(nBatch, Pi.size(0), Pi.size(1))
    q1 = qi.unsqueeze(0).expand(nBatch, qi.size(0))
    A1 = Ai.unsqueeze(0).expand(nBatch, Ai.size(0), Ai.size(1))
    b1 = bi.unsqueeze(0).expand(nBatch, bi.size(0))
    G1 = Gi.unsqueeze(0).expand(nBatch, Gi.size(0), Gi.size(1))
    h1 = hi.unsqueeze(0).expand(nBatch, hi.size(0))
    
    # x = QPFunction(eps=1e-3,verbose=False)(P1, q1, G1, h1, A1, b1)
    # x = QPFunction(eps=1e-3,verbose=False)(P1, q1, G1, h1, A1, b1)
    x = QPFunction(verbose=False)(P1, q1, G1, h1, A1, b1)
    # x = QPFunction(eps=1e-3,verbose=False)(P_np, q_np, G_np, h_np, A_np, b_np)
    x.sum().backward()
    b_h = bi.grad.cpu().numpy()
    

    x_opt = x.detach().T

    print("The optimal value is", (1/2) * x_opt.T @ Pi @ x_opt + qi.T @ x_opt)


    return (x_opt, b_h)

def diff(eps=1e-3, verbose=0):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, P_, q_, G_, h_, A_, b_):
            n, m, d = b_.shape[1], q_.shape[1], h_.shape[1]
            # print(n, m, d)
            P = decode(P_)
            q = q_.numpy()
            G = G_.numpy()
            h = h_.numpy()
            A = A_.numpy()
            b = b_.numpy()
            # Define and solve the CVXPY problem.
            optimal = []
            gradient = []

            for i in range(len(P)):
                begin = time.time()
                Pi, qi, Ai, bi, Gi, hi = P[i], q[i], A[i], b[i], G[i], h[i]

                xk, dxk = alt_diff(Pi, qi, Ai, bi, Gi, hi)

                end = time.time()
                optimal.append(xk)
                #print('iterations:', iters)
                gradient.append(dxk)

            ctx.save_for_backward(torch.tensor(np.array(gradient)))
            return torch.tensor((np.array(optimal)))

        @staticmethod
        def backward(ctx, grad_output):
            # only call parameters q
            grad = ctx.saved_tensors

            grad_all = torch.zeros((len(grad[0]),200))
            for i in range(len(grad[0])):
                grad_all[i] = grad_output[i] @ grad[0][i]
            #print(grad_all.shape)
            return (None, grad_all, None, None, None, None)

    return Newlayer.apply

