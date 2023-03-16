import numpy as np
import torch
import random

torch.set_default_dtype(torch.float64)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
thres = 1e-3



def param_random(n, m, p):
    np.random.seed(100)
    P = np.random.randn(n, n)
    P = torch.from_numpy(P.T @ P).to(device)
    q = torch.from_numpy(np.random.randn(n)).to(device)
    A = torch.from_numpy(np.random.randn(m, n)).to(device)
    b = torch.from_numpy(np.random.randn(m)).to(device).requires_grad_()

    G = torch.from_numpy(np.random.randn(p, n)).to(device)
    h = torch.from_numpy(np.random.randn(p)).to(device)

    return (P, q, A, b, G, h)


def sparseMatrix(M, dim1, dim2, rate):
    M = np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            if random.random() < rate:
                M[i][j] = random.random()
    return M

def param_sparse(n, m, p, rate):
    P = sparseMatrix(np.zeros((n,n)), n, n, rate)
    P = P.T @ P

    q = np.random.randn(n)

    A = sparseMatrix(np.zeros((m,n)), m, n, rate)
    #A = np.random.randn(m, n)
    b = np.random.randn(m).requires_grad_()

    G = sparseMatrix(np.zeros((p,n)), p, n, rate)
    #G = np.random.randn(p, n)
    h = np.random.randn(p)
    return (P, q, A, b, G, h)


def param_sparsemax(n):
    np.random.seed(100)

    P = torch.from_numpy(np.diag(np.ones(n))).to(device)

    y = np.random.randn(n)
    q = torch.from_numpy(-2 * y).to(device).requires_grad_()
    A = torch.from_numpy(np.random.randn(1, n)).to(device)
    #A = np.ones((1,n))
    b = torch.from_numpy(np.array([1])).to(device)

    G = torch.from_numpy(np.vstack((np.eye(n),-np.eye(n)))).to(device)
    h = torch.from_numpy(np.vstack((np.ones(n),np.zeros(n))).reshape(2 * n)).to(device)

    return (P, q, A, b, G, h)

def param_softmax(n, m, p):
    # np.random.seed(100)
    A = np.random.randn(m, n)
    #A = np.ones((1,n))
    #b = np.array(m)
    b = np.random.randn(m)
    #G = np.vstack((np.eye(n),-np.eye(n)))
    #h = np.vstack((np.ones(n),np.zeros(n))).reshape(2*n)
    
    G = np.random.randn(p,n)
    h = np.random.randn(p)

    y = np.random.rand(n)

    return (y, A, b, G, h)    