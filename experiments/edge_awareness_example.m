%%
% edge_awareness_example.m
%
%%

%% Setup graphs
G1 = graph([1 2 3 4 5 6 7 8],[2 3 4 5 6 7 8 1]);
d1 = zeros(8,2);
for i=1:8
    d1(i,1) = cos(pi/8+pi/4*(i-1));
    d1(i,2) = sin(pi/8+pi/4*(i-1));
end

G2 = graph([1 2 3 4 5 6 7],[2 3 4 5 6 7 8]);
d2 = d1;

G3 = graph([1 2 3 4 5 6 7],[2 3 4 5 6 7 8]);
d3 = zeros(8,2);
for i=1:8
    d3(i,1) = cos(pi/2+pi/7*(i-1));
    d3(i,2) = sin(pi/2+pi/7*(i-1));
end



%% Do NetOTC
% Get Adjacency matrix
A1 = adjacency(G1);
A1 = full(A1);

A2 = adjacency(G2);
A2 = full(A2);

A3 = adjacency(G3);
A3 = full(A3);


% Numbers of nodes of G1, G2, G3
n = 8;

% Get transition matrices
P1 = A1 ./ sum(A1, 2);
P2 = A2 ./ sum(A2, 2);
P3 = A3 ./ sum(A3, 2);

% Get distributions 
stat_dist1 = approx_stat_dist(P1, 100)';
stat_dist2 = approx_stat_dist(P2, 100)';
stat_dist3 = approx_stat_dist(P3, 100)';

unif_dist1 = ones(n,1)/n;
unif_dist2 = ones(n,1)/n;
unif_dist3 = ones(n,1)/n;

% Get cost matrix
c1 = zeros([n, n]);
for i=1:n
    for j=1:n
        c1(i, j) = sum((d2(i,:)-d1(j,:)).^2);
    end
end

c2 = zeros([n, n]);
for i=1:n
    for j=1:n
        c2(i, j) = sum((d2(i,:)-d3(j,:)).^2);
    end
end


% Run algorithm
% G2 vs G1
[otc_distance1, gotc1, otc_alignment1] = exact_otc(P2, P1, c1);
otc_distance1

[~, otsd_distance1] = computeot_lp(c1', stat_dist2, stat_dist1');
otsd_distance1

[fgw_distance1, fgw_alignment1] = fgw_dist(c1, A2, A1, unif_dist2, unif_dist1, 1, 0.5);
fgw_distance1

% G2 vs G3
[otc_distance2, gotc2, otc_alignment2] = exact_otc(P2, P3, c2);
otc_distance2

[~, otsd_distance2] = computeot_lp(c2', stat_dist2, stat_dist3');
otsd_distance2

[fgw_distance2, fgw_alignment2] = fgw_dist(c2, A2, A3, unif_dist2, unif_dist3, 1, 0.5);
fgw_distance2

%% Plot graphs G1, G2, G3
format longG;
savedir = ['./plot/'];
mkdir(savedir);

plot(G1,'XData',d1(:,1),'YData',d1(:,2),'LineWidth',6,'NodeFontSize',16) %'NodeColor','k','EdgeColor','k')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
grid on
axis square
saveas(gcf,[savedir 'circle_G1.png'])


plot(G2,'XData',d2(:,1),'YData',d2(:,2),'LineWidth',6,'NodeFontSize',16) %'NodeColor','k','EdgeColor','k')
xlim([-1.2 1.2])
ylim([-1.2 1.2])
grid on
axis square
saveas(gcf,[savedir 'circle_G2.png'])

plot(G3,'XData',d3(:,1),'YData',d3(:,2),'LineWidth',6,'NodeFontSize',16)
xlim([-1.2 1.2])
ylim([-1.2 1.2])
grid on
axis square
saveas(gcf,[savedir 'circle_G3.png'])

function dist = approx_stat_dist(P, iter)
    n = size(P, 1);
    dist = zeros(1, n);
    dist(1) = 1;
    for i=1:iter
        dist = dist*P;
    end
end

function [lp_sol,lp_val] = computeot_lp( C,r,c )
% vectorize P and C by: column 1, column 2, etc.
nx = size(r, 1);
ny = size(c, 2);
Aeq = zeros(nx+ny,nx*ny);
beq = [r;c'];

% column sums correct
for row=1:nx
    for t=1:ny
        Aeq(row,(row-1)*ny+t)=1;
    end
end

% row sums correct
for row=nx+1:nx+ny
    for t=0:nx-1
        Aeq(row,(row-nx)+t*ny) = 1;
    end
end

% ensure positivity of each entry
lb = zeros(nx*ny,1);

% solve OT LP using linprog
cost = reshape(C,nx*ny,1);
options = optimoptions('linprog','Display','none');
[lp_sol,lp_val] = linprog(cost,[],[],Aeq,beq,lb,[],options);
end

function [exp_cost, P, stat_dist] = exact_otc(Px, Py, c)
dx = size(Px, 1);
dy = size(Py, 1);

P_old = ones(dx*dy);
P = get_ind_tc(Px, Py);
iter_ctr = 0;
while max(max(abs(P-P_old))) > 1e-10
    iter_ctr = iter_ctr + 1;
    P_old = P;
    
    % Transition coupling evaluation.
    [g, h] = exact_tce(P, c);
    
    % Transition coupling improvement.
    P = exact_tci(g, h, P_old, Px, Py);
    
    % Check for convergence.
    if all(all(P == P_old))
        [stat_dist, exp_cost] = get_best_stat_dist(P, c);
        stat_dist = reshape(stat_dist, dy, dx)';
        return
    end
end
end

%%
% get_ind_tc.m
%
% Compute independent transition coupling of two transition matrices.

function [P_ind] = get_ind_tc(Px, Py)
    [dx, dx_col] = size(Px);
    [dy, dy_col] = size(Py);
    
    P_ind = zeros(dx*dy, dx_col*dy_col);
    for x_row=1:dx
        for x_col=1:dx_col
            for y_row=1:dy
                for y_col=1:dy_col
                    idx1 = dy*(x_row-1)+y_row;
                    idx2 = dy*(x_col-1)+y_col;
                    P_ind(idx1, idx2) = Px(x_row, x_col)*Py(y_row, y_col);
                end
            end
        end
    end
end

%%
% exact_tce.m
%
% Exact transition coupling evaluation.

function [g, h] = exact_tce(P, c)
d = size(P, 1);
c = reshape(c', d, []);
A = [eye(d)-P, zeros(d), zeros(d)
    eye(d), eye(d)-P, zeros(d)
    zeros(d), eye(d), eye(d)-P];
b = [zeros(d, 1); c; zeros(d, 1)];
[sol, r] = linsolve(A, b);
% If the linear solver failed. Use pseudoinverse.
if r == 0
    sol = pinv(A)*b;
end
sol = sol';
g = sol(1:d)';
h = sol((d+1):(2*d))';
end

%%
% exact_tci.m
%
% Exact transition coupling improvement.

function P = exact_tci(g, h, P0, Px, Py)
x_sizes = size(Px);
y_sizes = size(Py);
dx = x_sizes(1);
dy = y_sizes(1);
P = zeros(dx*dy, dx*dy);
%% Try to improve with respect to g.
% Check if g is constant.
g_const = 1;
for i=1:dx
    for j=(i+1):dy
        if abs(g(i) - g(j)) > 1e-3
            g_const = 0;
            break
        end
    end
end
% If g is not constant, improve transition coupling against g.
if ~g_const
    g_mat = reshape(g, dy, dx)';
    for x_row=1:dx
        for y_row=1:dy
           dist_x = Px(x_row,:);
           dist_y = Py(y_row,:);
           % Check if either distribution is degenerate.
           if any(dist_x == 1) | any(dist_y == 1)
               sol = dist_x' * dist_y;
           % If not degenerate, proceed with OT.
           else
               [sol, val] = computeot_lp(g_mat', dist_x', dist_y);
           end
           idx = dy*(x_row-1)+y_row;
           P(idx,:) = reshape(sol', [], dx*dy);
        end
    end
    if max(abs(P0*g - P*g)) <= 1e-7
        P = P0;
    else
        return
    end
end
    
%% Try to improve with respect to h.
h_mat = reshape(h, dy, dx)';
for x_row=1:dx
    for y_row=1:dy
       dist_x = Px(x_row,:);
       dist_y = Py(y_row,:);
       % Check if either distribution is degenerate.
       if any(dist_x == 1) | any(dist_y == 1)
           sol = dist_x' * dist_y;
       % If not degenerate, proceed with OT.
       else
           [sol, val] = computeot_lp(h_mat', dist_x', dist_y);
       end
       idx = dy*(x_row-1)+y_row;       
       P(idx,:) = reshape(sol', [], dx*dy);
    end
end
if max(abs(P0*h - P*h)) <= 1e-4
    P = P0;
end
end

%%
% get_best_stat_dist.m
%
% Compute best stationary distribution of a transition matrix given
% a cost matrix c. 
%
% Inputs:
% -P: transition matrix in R^(n x n)
% -c: cost matrix in R^(n x n)
%
% Outputs:
% -stat_dist: vector in R^n corresponding to best stationary distribution
% of P with respect to c.
% -exp_cost: the expected cost of stat_dist with respect to c, computed by
% simply taking the inner product of stat_dist and c.

function [stat_dist, exp_cost] = get_best_stat_dist(P, c)
    % Set up constraints.
    n = size(P,1);
    c = reshape(c', n, []);
    Aeq = [P' - eye(n);
           ones(1, n)];
    beq = [zeros(n, 1);
           1];
    lb = zeros(n,1);
    % Solve linear program.
    options = optimset('Display','off', 'TolCon', 1e-3, 'TolFun', 1e-6);
    options.Preprocess = 'none';
    %options = optimset('Display','off');
    [stat_dist, exp_cost] = linprog(c, [], [], Aeq, beq, lb, [], options);
    
    % In case the solver fails due to numerical underflow, try with
    % rescaling.
    alpha = 1;
    while isempty(stat_dist) & alpha >= 1e-10
        alpha = alpha/10;
        [stat_dist, exp_cost] = linprog(c, [], [], alpha*Aeq, alpha*beq, lb, [], options);
    end
    if isempty(stat_dist)
        error('Failed to compute stationary distribution.');
    end
end

%%
%   Fused Gromov-Wasserstein distance.
%

function [FGW,pi] = fgw_dist(M, C1, C2, mu1, mu2, q, alpha)

    % Define some helper functions.
    function loss = fgw_loss(pi)
        loss = (1-alpha)*sum(sum(M.^q.*pi));
        m = size(M,1);
        n = size(M,2);
        for i=1:m
            for j=1:n
                for k=1:m
                    for l=1:n
                        loss = loss + 2*alpha*abs(C1(i,k)-C2(j,l))^q*pi(i,j)*pi(k,l);
                    end
                end
            end
        end
    end

    function grad = fgw_grad(pi)
        grad = (1-alpha)*(M.^q);
        m = size(M,1);
        n = size(M,2);
        for i=1:m
            for j=1:n
                for k=1:m
                    for l=1:n
                        grad(i,j) = grad(i,j) + 2*alpha*abs(C1(i,k)-C2(j,l))^q*pi(k,l);
                    end
                end
            end
        end
    end

    % Initialize coupling
    pi = mu1 .* mu2';
    m = size(pi,1);
    n = size(pi,2);
    
    % Run algorithm
    n_iter = 100;
    for iter=1:n_iter
       % Compute gradient
       G = fgw_grad(pi);
       % Solve OT problem with cost G
       [pi_new, ~] = computeot_lp(G', mu1, mu2');
       pi_new = reshape(pi_new',n,m)';
       % Line search
       fun = @(tau) (fgw_loss((1-tau)*pi+tau*pi_new));
       %tau = fminbnd(fun,0,1);
       tau_vec = 0:0.1:1;
       [~,tau_idx] = min(arrayfun(fun, tau_vec));
       tau = tau_vec(tau_idx);
       % Store updated coupling
       pi = (1-tau)*pi + tau*pi_new;
    end
    
    % Store result
    FGW = fgw_loss(pi);
end