import numpy as np
from scipy.optimize import linprog

def approx_stat_dist(P, iter):
    n = P.shape[0]
    dist = np.zeros(n)
    dist[0] = 1
    for i in range(iter):
        dist = np.dot(dist, P)
    return dist

def get_ind_tc(Px, Py):
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape
    P_ind = np.zeros((dx * dy, dx_col * dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy * (x_row) + y_row
                    idx2 = dy * (x_col) + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
    return P_ind

def exact_tce(P, c):
    d = P.shape[0]
    c = c.reshape((d, -1))
    A = np.block([[np.eye(d) - P, np.zeros((d, d)), np.zeros((d, d))],
                  [np.eye(d), np.eye(d) - P, np.zeros((d, d))],
                  [np.zeros((d, d)), np.eye(d), np.eye(d) - P]])
    b = np.concatenate([np.zeros(d), c.flatten(), np.zeros(d)])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    g = sol[:d]
    h = sol[d:2*d]
    return g, h

def computeot_lp(C, r, c):
    nx = len(r)
    ny = len(c)
    A_eq = np.zeros((nx + ny, nx * ny))
    b_eq = np.concatenate([r, c])

    for row in range(nx):
        for t in range(ny):
            A_eq[row, row * ny + t] = 1

    for row in range(nx, nx + ny):
        for t in range(nx):
            A_eq[row, (row - nx) + t * ny] = 1

    lb = np.zeros(nx * ny)

    cost = C.reshape(nx * ny)
    options = {'disp': False, 'tol': 1e-9, 'presolve': False, 'method': 'interior-point'}
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None) for _ in range(nx * ny)], options=options)

    return res.x, res.fun
def exact_tci(g, h, P0, Px, Py):
    x_sizes = Px.shape
    y_sizes = Py.shape
    dx = x_sizes[0]
    dy = y_sizes[0]
    P = np.zeros((dx * dy, dx * dy))
    
    # Try to improve with respect to g
    g_const = np.all(np.isclose(g, g[0], atol=1e-3))
    
    if not g_const:
        g_mat = g.reshape((dy, dx)).T
        for x_row in range(dx):
            for y_row in range(dy):
                dist_x = Px[x_row, :]
                dist_y = Py[y_row, :]
                if np.any(dist_x == 1) or np.any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y).flatten()
                else:
                    sol, _ = computeot_lp(g_mat.T.flatten(), dist_x, dist_y)
                idx = dy * (x_row) + y_row
                P[idx, :] = sol.reshape((-1, dx * dy))
        if np.max(np.abs(np.dot(P0, g) - np.dot(P, g))) <= 1e-7:
            P = P0

    # Try to improve with respect to h
    h_mat = h.reshape((dy, dx)).T
    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]
            if np.any(dist_x == 1) or np.any(dist_y == 1):
                sol = np.outer(dist_x, dist_y).flatten()
            else:
                sol, _ = computeot_lp(h_mat.T.flatten(), dist_x, dist_y)
            idx = dy * (x_row) + y_row
            P[idx, :] = sol.reshape((-1, dx * dy))
            
    if np.max(np.abs(np.dot(P0, h) - np.dot(P, h))) <= 1e-4:
        P = P0
    
    return P

def get_best_stat_dist(P, c):
    """
    Compute best stationary distribution of a transition matrix given
    a cost matrix c.

    Parameters:
    P (numpy.ndarray): transition matrix of shape (n, n)
    c (numpy.ndarray): cost matrix of shape (n, n)

    Returns:
    tuple: (stat_dist, exp_cost)
        stat_dist (numpy.ndarray): vector of shape (n,) corresponding to best stationary distribution of P with respect to c.
        exp_cost (float): the expected cost of stat_dist with respect to c.
    """
    # Set up constraints.
    n = P.shape[0]
    c = c.ravel()  # Reshape to a 1D array to match Matlab code
    Aeq = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    beq = np.zeros(n + 1)
    beq[-1] = 1
    # lb = np.zeros(n)
    
    # Solve linear program.
    options = {'disp': False, 'tol': 1e-6, 'presolve': False}
    # checked
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None) for _ in range(n)], options=options)
    
    # stat_dist,exp_cost = pulp_lp(c, Aeq, beq)
    stat_dist = res.x
    exp_cost = res.fun
    return stat_dist, exp_cost

def exact_otc(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]
    
    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py) # checked
    # print(f'P: {P}')
    iter_ctr = 0
    while np.max(np.abs(P - P_old)) > 1e-10:
        iter_ctr += 1
        P_old = P
        g, h = exact_tce(P, c)
        # The following line is a placeholder. The function exact_tci needs to be implemented.
        P = exact_tci(g, h, P_old, Px, Py) # checked
        # Check for convergence (placeholder)
        if np.all(P == P_old):
            # P, c Checked
            stat_dist, exp_cost = get_best_stat_dist(P, c)
            
            return exp_cost, P, stat_dist.reshape((dy, dx), order='F').T # 使用列优先（column-major）顺序 reshape

def fgw_dist(M, C1, C2, mu1, mu2, q, alpha):
    def fgw_loss(pi):
        loss = (1 - alpha) * np.sum(M ** q * pi)
        m, n = pi.shape
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        loss += 2 * alpha * abs(C1[i, k] - C2[j, l]) ** q * pi[i, j] * pi[k, l]
        return loss

    def fgw_grad(pi):
        grad = (1 - alpha) * (M ** q)
        m, n = pi.shape
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        grad[i, j] += 2 * alpha * abs(C1[i, k] - C2[j, l]) ** q * pi[k, l]
        return grad

    pi = (mu1[:, None] * mu2[None, :])[:,:,0]
    m, n = pi.shape

    n_iter = 100
    for iter in range(n_iter):
        G = fgw_grad(pi)
        pi_new, _ = computeot_lp(G.flatten(), mu1, mu2)
        pi_new = pi_new.reshape((n, m)).T

        fun = lambda tau: fgw_loss((1 - tau) * pi + tau * pi_new)
        tau_values = np.linspace(0, 1, 11)
        tau = tau_values[np.argmin([fun(t) for t in tau_values])]

        pi = (1 - tau) * pi + tau * pi_new

    FGW = fgw_loss(pi)
    return FGW, pi

def eval_alignment(coupling, block_size, n_blocks):
    alignment = 0
    for b in range(n_blocks):
        for i in range(block_size):
            idx = b * block_size + i
            alignment += coupling[idx, b]
    return alignment
