from scipy.optimize import *
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import math
import time
# from cvxpy.atoms.affine.wraps import psd_wrap
import torch
from cvxpylayers.torch import CvxpyLayer
import osqp
from util import cosDis, relu, sgn

def f(x):
    return -y.T @ x + x @ np.log(x)

def test_bfgs_Alt_diff(n, m, p, A, b, G, h, y, thres):
    sk = np.zeros(p)
    lamb = np.zeros(m)
    nu = np.zeros(p)

    dxk = np.zeros((m, n))
    dsk = np.zeros((p, n))
    dlamb = np.zeros((m, n))
    dnu = np.zeros((p, n))

    rho = 0.1

    coef = rho * A.T @ A + rho * G.T @ G

    res = [1000, -100]
    # thres = 1e-5

    begin2 = time.time()
    xk = np.ones(n)
    grad1 = np.zeros((n, n))

    eta = 0.5
    for _ in range(50):
        res_inner = [1000, -100]

        while abs((res_inner[-1] - res_inner[-2]) / res_inner[-2]) > thres:
            inv = np.diag(1 / xk)

            hessian_inv = np.linalg.inv(inv + coef)

            grad = - y + np.ones(n) + np.log(xk) + A.T @ lamb + G.T @ nu + rho * A.T @ (A @ xk - b) + rho * G.T @ (
                    G @ xk + sk - h)

            xk -= eta * hessian_inv @ grad

            res_inner.append(-y.T @ xk + xk @ np.log(xk))

        dxk = - hessian_inv @ (-np.ones(n) + A.T @ dlamb + G.T @ dnu + rho * G.T @ dsk)

        sk = relu(- (1 / rho) * nu - (G @ xk - h))

        dsk = (-1 / rho) * sgn(sk).reshape(p, 1) @ np.ones((1, n)) * (dnu + rho * G @ dxk)
        dsk = dsk.numpy()

        lamb = lamb + rho * (A @ xk - b)

        dlamb = dlamb + rho * (A @ dxk)

        nu = nu + rho * (G @ xk + sk - h)

        dnu = dnu + rho * (G @ dxk + dsk)

        res.append(-y.T @ xk + xk @ np.log(xk))

    y_f = dxk.T[0]
    end2 = time.time()

    return xk, y_f, end2 - begin2, len(res) - 2

def test_bfgs_cvxpylayers(n, A, b, G, h, y, thres):
    begin1 = time.time()
    y0 = cp.Parameter(n)
    x = cp.Variable(n)
    obj = cp.Minimize(-y0 @ x - cp.sum(cp.entr(x)))
    constraint = [A @ x == b, G @ x <= h]
    prob = cp.Problem(obj, constraint)

    begin0 = time.time()
    layer = CvxpyLayer(prob, parameters=[y0], variables=[x])
    y_tch = torch.from_numpy(y).requires_grad_()

    print('Initialization time is', time.time() - begin0)

    for i in range(1):
        print(f'(trial {i})')
        begin1 = time.time()

        solution, = layer(y_tch, solver_args={'mode': 'lsqr', 'eps': thres})
        solution.sum().backward()
        y_g = y_tch.grad.numpy()
        x_opt = solution.detach().numpy()
        end1 = time.time()

    return x_opt, y_g, end1 - begin1

if __name__ == '__main__':
    # n, m, p = 10000, 3000, 1000
    # n, m, p = 5000, 2000, 1000
    # n, m, p = 3000, 1000, 500
    n, m, p = 1000, 300, 100

    # np.random.seed(100)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(p, n)
    h = np.random.randn(p)
    y = np.random.rand(n)

    print('--------cvxpylayers---------')
    x_opt, y_g, time_cvxpylayers = test_bfgs_cvxpylayers(n, A, b, G, h, y, thres=1e-3)
    print('The running time in', n, 'dim is', time_cvxpylayers)

    print('\n--------Alt-Diff, full ---------')
    xk, y_f, time_alt_diff, iter_alt_diff = test_bfgs_Alt_diff(n, m, p, A, b, G, h, y, thres=1e-5)
    # print("The optimal value is", (1/2) * xk.T @ P @ xk + q.T @ xk)
    # print("A solution x is", xk)
    print("The running time is", time_alt_diff)
    print('num iter: ', iter_alt_diff)

    print("The cosine similarity in x between Alt-Diff and cvxpy is", cosDis(xk, x_opt))
    print("The cosine similarity in grad between Alt-Diff and cvxpy is", cosDis(y_f, y_g))



