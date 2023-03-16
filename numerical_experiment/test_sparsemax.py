import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from cvxpy.atoms.affine.wraps import psd_wrap
import torch
from cvxpylayers.torch import CvxpyLayer
from qpth.qp import QPFunction, QPSolvers
import scipy

def cosDis(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


thres = 1e-3

n = 1000
print(f'Dimension is {n}')
np.random.seed(100)

P = np.diag(np.ones(n))

y = np.random.randn(n)
q = -2 * y
A = np.random.randn(1, n)
#A = np.ones((1,n))
b = np.array([1])

G = np.vstack((np.eye(n),-np.eye(n)))
h = np.vstack((np.ones(n),np.zeros(n))).reshape(2*n)

print("-----------optnet-------------")
for i in range(1):
    print(f'(trial {i})')
    beginopt = time.time()
    nBatch = 1
    with torch.no_grad():
        P_tch = torch.from_numpy(P)
        y_tch = torch.from_numpy(y).requires_grad_()
        q_tch = torch.from_numpy(q).requires_grad_()
        A_tch = torch.from_numpy(A)
        b_tch = torch.from_numpy(b)
        G_tch = torch.from_numpy(G)
        h_tch = torch.from_numpy(h)
    
#    b_tch = torch.from_numpy(b).requires_grad_()
    P1 = P_tch.unsqueeze(0).expand(nBatch, P_tch.size(0), P_tch.size(1))
    q1 = q_tch.unsqueeze(0).expand(nBatch, q_tch.size(0))
    A1 = A_tch.unsqueeze(0).expand(nBatch, A_tch.size(0), A_tch.size(1))
    b1 = b_tch.unsqueeze(0).expand(nBatch, b_tch.size(0))
    G1 = G_tch.unsqueeze(0).expand(nBatch, G_tch.size(0), G_tch.size(1))
    h1 = h_tch.unsqueeze(0).expand(nBatch, h_tch.size(0))
    
    #x = QPFunction(eps = thres, verbose=False)(P1, q1, G1, h1, A1, b1)
    x = QPFunction(eps = thres, verbose=False)(P1, q1, G1, h1, A1, b1)
    x.sum().backward()
    q_h = q_tch.grad.numpy()
        
    endopt = time.time()
    x = x.detach().numpy().T
    #print(x.shape)
    print('The optimal value is', ((1/2) * x.T @ P @ x + q.T @ x)[0][0])
    
    print("The running time is", endopt-beginopt)

time0 = time.time()

q_tch = torch.from_numpy(q).requires_grad_()

q0 = cp.Parameter(n)
q0.value = q
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, psd_wrap(P)) + q0.T @ x),
                 [A @ x == b, G @ x <= h])
#prob.solve()
#print(prob.value)
print('--------cvxpylayers---------')

layer = CvxpyLayer(prob, parameters=[q0], variables=[x])
print('Initialization time is', time.time()-time0)
#print(layer)

for i in range(1):
    print(f'(trial {i})')
    begin1 = time.time()
    # solution, = layer(q_tch, solver_args={'mode': 'lsqr', 'eps': thres})
    solution, = layer(q_tch, solver_args={'mode': 'lsqr'})
    solution.sum().backward()
    q_g = q_tch.grad.numpy()
    end1 = time.time()
 
    x_opt = solution.detach().numpy()
 
    print("The optimal value is", (1/2) * x_opt.T @ P @ x_opt + q.T @ x_opt)
    #print("A solution x is", x.value)
    print("The running time is", end1-begin1)
#    print("Solver time is", prob.solver_stats.solve_time)
    print("The total running time is", end1-time0)

p = 2 * n
m = 1


print('\n--------Alt-Diff, full ---------')
def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i]<0:
            ss[i] = 0
    return ss
 
def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i]<0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001))/2
    return ss
 
def sgn(s):
    ss = np.zeros(p)
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss


xk = np.zeros(n)
sk = np.zeros(p)
lamb_all = []
nu_all = []
lamb = np.zeros(m)
nu = np.zeros(p)
 

dxk = np.zeros((m,n))
dsk = np.zeros((p,m))
dlamb = np.zeros((m,m))
dnu = np.zeros((p,m))
 
rho = 1
M = P + rho * A.T @ A + rho * G.T @ G

begininv = time.time()
R = - np.linalg.inv(P + rho * A.T @ A + rho * G.T @ G)
endinv = time.time()
print("The inverse time is", endinv-begininv)


res = [1000,-100]
thres = 1e-4
M_csr = scipy.sparse.csr_matrix(M)
begin2 = time.time()
#while abs((res[-1]-res[-2])/res[-2])>thres:
while abs((np.linalg.norm(res[-1]) - np.linalg.norm(res[-2])) / np.linalg.norm(res[-2])) > thres:
    #coef2 = q + A.T @ lamb + G.T @ nu - rho * A.T @ b + rho * G.T @ (sk - h)
    #xk = scipy.sparse.linalg.lsqr(M_csr, coef2)[0]
    xk = R @ (q + A.T @ lamb + G.T @ nu - rho * A.T @ b + rho * G.T @ (sk - h))
    dxk = R @ (np.eye(m) + A.T @ dlamb + G.T @ dnu  + rho * G.T @ dsk)
 
    sk = relu(- (1 / rho) * nu - (G @ xk - h))
    dsk = (-1 / rho) * sgn(sk).reshape(p,1) @ np.ones((1,m)) * (dnu + rho * G @ dxk)
 
    lamb = lamb + rho * (A @ xk - b)
    lamb_all.append(lamb)
    dlamb = dlamb + rho * (A @ dxk)
 
    nu = nu + rho * (G @ xk + sk - h)
    nu_all.append(nu)
    dnu = dnu + rho * (G @ dxk + dsk)
    res.append(xk)
    #res.append(0.5 * (xk.T @ P @ xk) + q.T @ xk)
 
#q_f = np.sum(dxk,axis=0)
q_f = dxk.T[0]
end2 = time.time()
 

 
print("The optimal value is", (1/2) * xk.T @ P @ xk + q.T @ xk)
#print("A solution x is", xk)
print("The running time is", end2-begin2)
print('num iter: ', len(res))
 
print("The cosine similarity between Alt-Diff and cvxpy is", cosDis(q_f,q_g))
print("The cosine similarity between Alt-Diff and optnet is", cosDis(q_f,q_h))
#print(P.T @ dxk + A.T @ dlamb + G.T @ dnu)
#print(A @ dxk - np.eye(m))
#print(nu @ G @ dxk - (G @ xk - h) @ dnu)
print('The total running time is', end2-begininv)


print('\n-------- Alt-Diff (grad only) ---------')
 
dxk = np.zeros((m,n))
dsk = np.zeros((p,m))
dlamb = np.zeros((m,m))
dnu = np.zeros((p,m))
 
rho = 1

begininv = time.time()
R = - np.linalg.inv(P + rho * A.T @ A + rho * G.T @ G)
endinv = time.time()
print("The inverse time is", endinv-begininv)

 
res1 = [1000,-100]

begin3 = time.time()
n_backward_iter = 0
#for i in range(30):
while abs((res1[-1]-res1[-2])/res1[-2])>thres:    
    
    n_backward_iter += 1
    dxk = R @ (np.eye(m) + A.T @ dlamb + G.T @ dnu  + rho * G.T @ dsk)
    dsk = (-1 / rho) * sgn(sk).reshape(p,1) @ np.ones((1,m)) * (dnu + rho * G @ dxk)
    dlamb = dlamb + rho * (A @ dxk - np.eye(m))
    dnu = dnu + rho * (G @ dxk + dsk)
    res1.append(dxk.sum())
    b_f = dxk

end3 = time.time()
print('num iter: ', n_backward_iter)
print("The running time is", end3-begin3)
print("The cosine similarity between Alt-Diff and cvxpy is", cosDis(b_f.reshape(n),q_g))
print("The cosine similarity between Alt-Diff and optnet is", cosDis(b_f.reshape(n),q_h))
print('The total running time is', end3-begininv)
