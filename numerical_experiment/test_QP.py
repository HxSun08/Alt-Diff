from data_generation import param_random, param_sparsemax
from opt_layer import alt_diff, cvxpylayer, optnet
from util import *
import time

def test_QP_Alt_Diff(n, m, p):
    P, q, A, b, G, h = param_random(n, m, p)
    for i in range(1):
        begin_alt = time.time()
        print(f'(trial {i})')
        xk, dxk = alt_diff(P, q, A, b, G, h)
        # print((1 / 2) * xk.T @ P @ xk + q.T @ xk)
        b_f = torch.sum(dxk, axis=0)
        end_alt = time.time()
        return b_f, end_alt - begin_alt

def test_QP_optnet(n, m, p):
    P, q, A, b, G, h = param_random(n, m, p)
    for i in range(1):
        begin_opt = time.time()
        print(f'(trial {i})')
        x_opt, b_h = optnet(P, q, A, b, G, h)
        end_opt = time.time()
        return b_h, end_opt - begin_opt

def test_QP_cvxpylayers(n, m, p):
    P, q, A, b, G, h = param_random(n, m, p)
    for i in range(1):
        begin_cvx = time.time()
        print(f'(trial {i})')
        x_cvx, b_g = cvxpylayer(P, q, A, b, G, h)
        end_cvx = time.time()
        return b_g, end_cvx - begin_cvx

if __name__ == '__main__':
    # n, m, p = 10000, 5000, 2000
    # n, m, p = 5000, 2000, 1000
    n, m, p = 3000, 1000, 500
    # n, m, p = 1000, 500, 200

    b_f, Alt_Diff = test_QP_Alt_Diff(n, m, p)
    print("The running time is of Alt-Diff is", Alt_Diff)

    b_h, optnet = test_QP_optnet(n, m, p)
    print("The running time is of optnet is", optnet)

    print("Cosine distance between Alt-Diff and optnet is ", cosDis(b_h, b_f.detach().cpu().numpy()))

    b_g, cvxpylayer = test_QP_cvxpylayers(n, m, p)
    print("The running time is of cvxpy is", cvxpylayer)

    print("Cosine distance between Alt-Diff and cvxpy is ", cosDis(b_g, b_f.detach().cpu().numpy()))



