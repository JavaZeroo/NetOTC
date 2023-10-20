% Network Factors Experiment

% Seed number
%rng(311);

% Params
n_iter = 1; %100
block_size = 5;
n_blocks = 2;
n = block_size*n_blocks;
dim = 2;

for mean1=1:1 %1:4
    values = table([], []);
    values.Properties.VariableNames = {'Algorithm' 'Accuracy'};
    mean_sigma = 0.5 + mean1 * 0.5;
    disp(["mean_sigma ", num2str(mean_sigma)]);
    for iter=1:n_iter

        %% Setup graph
        % Sample vertices of G1 and G2
        [V1, V2] = get_pointcloud(mean_sigma, block_size, n_blocks, dim);


        %disp(V1)
        %disp(V2)
        %A2 = randi(10, n_blocks, n_blocks);
        A2 = [
            5, 7;
            3, 3
        ];

        A1 = zeros(n,n);
        for i = 1:n_blocks
            for j = 1:n_blocks
                for k = 1:block_size
                    rv = rand(1,block_size);
                    rv = rv * A2(i,j) / sum(rv) / block_size;
                    A1((i-1)*block_size+k, (j-1)*block_size+1:j*block_size) = rv;
                end
            end
        end
        V1 = [
            1.4786, -0.2885;
           -0.8243, -1.7479;
           -0.5537, -0.1875;
           -0.4139, -0.5610;
            0.1822, -0.1728;
           -0.7730, -0.5754;
           -0.6185,  0.3518;
           -0.4191, -0.7918;
           -0.4780, -0.1640;
           -1.6767, -0.6409
        ];
        
        V2 = [
            0.2192, -0.8808;
           -0.4677, -0.2467
        ];
        A1 = [
            0.3732, 0.2756, 0.1131, 0.2063, 0.0318, 0.3099, 0.1997, 0.3172, 0.1452, 0.4280;
            0.0380, 0.3102, 0.2640, 0.3277, 0.0602, 0.2431, 0.3631, 0.2348, 0.2918, 0.2672;
            0.2087, 0.1455, 0.2907, 0.1457, 0.2094, 0.4533, 0.3134, 0.5515, 0.0531, 0.0287;
            0.5610, 0.1954, 0.0877, 0.0836, 0.0723, 0.0174, 0.2860, 0.2580, 0.3980, 0.4406;
            0.1354, 0.2900, 0.1278, 0.1819, 0.2650, 0.6785, 0.5636, 0.0525, 0.0055, 0.0999;
            0.0898, 0.0852, 0.1621, 0.1234, 0.1396, 0.1554, 0.1405, 0.0924, 0.1292, 0.0825;
            0.0245, 0.0141, 0.0405, 0.2885, 0.2324, 0.1785, 0.1503, 0.0545, 0.0357, 0.1811;
            0.0221, 0.1822, 0.1985, 0.0923, 0.1049, 0.0715, 0.1978, 0.1074, 0.1214, 0.1019;
            0.2546, 0.0795, 0.0283, 0.1815, 0.0561, 0.2094, 0.1357, 0.0052, 0.2314, 0.0184;
            0.1280, 0.0451, 0.3293, 0.0773, 0.0203, 0.1448, 0.3196, 0.0410, 0.0077, 0.0869
        ];


        %% Do GraphOTC
        % Get transition matrices

        P1 = A1 ./ sum(A1, 2);
        P2 = A2 ./ sum(A2, 2);
        %disp(P1)
        stat_dist1 = approx_stat_dist(P1, 100)';
        stat_dist2 = approx_stat_dist(P2, 100)';
        %disp(stat_dist1)
        stat_dist3 = ones(n,1)/n;
        stat_dist4 = ones(n_blocks,1)/n_blocks;
        %disp(size(stat_dist3))
        %disp(size(stat_dist4))
        %disp('==========================')
        % Get cost matrix
        c = zeros([n, n_blocks]);
        for i=1:n
            for j=1:n_blocks
                c(i, j) = sum((V1(i,:)-V2(j,:)).^2);
            end
        end
        


        
        % Run algorithms
        [cost, otc_edge_alignment, otc_alignment] = exact_otc(P1, P2, c);
        disp(cost)
        [~, fgw_alignment] = fgw_dist(c, A1, A2, stat_dist3, stat_dist4, 1, 0.5);
        [otsd_alignment, ~] = computeot_lp(c', stat_dist1, stat_dist2');
        otsd_alignment = reshape(otsd_alignment, n_blocks, n)';

        aligned_mass_otc = eval_alignment(otc_alignment, block_size, n_blocks);
        aligned_mass_fgw = eval_alignment(fgw_alignment, block_size, n_blocks);
        aligned_mass_otsd = eval_alignment(otsd_alignment, block_size, n_blocks);
        disp(aligned_mass_otc)
        % Store results
        values = [values; {'OTC' aligned_mass_otc}];
        values = [values; {'FGW' aligned_mass_fgw}];          
        values = [values; {'OT-SD' aligned_mass_otsd}];
    end

    disp(['OTC mean accuracy: ' num2str(mean(values{strcmp(values{:,1}, 'OTC'),2}))]);
    disp(['FGW mean accuracy: ' num2str(mean(values{strcmp(values{:,1}, 'FGW'),2}))]);
    disp(['OT-SD mean accuracy: ' num2str(mean(values{strcmp(values{:,1}, 'OT-SD'),2}))]);

    disp(['OTC accuracy standard deviation: ' num2str(std(values{strcmp(values{:,1}, 'OTC'),2}))]);
    disp(['FGW accuracy standard deviation: ' num2str(std(values{strcmp(values{:,1}, 'FGW'),2}))]);
    disp(['OT-SD accuracy standard deviation: ' num2str(std(values{strcmp(values{:,1}, 'OT-SD'),2}))]);
end



%% Helper functions
% Point cloud sampling function
function [points, factor] = get_pointcloud(mean_sigma, n_points, n_clouds, dimension)
    point_sigma = 1;
    
    % Randomly draw means of each Gaussian
    means = normrnd(0, mean_sigma, n_clouds, dimension);
    factor = means;
    
    % Draw point clouds
    points = zeros(n_points*n_clouds, dimension);
    for c = 1:n_clouds
        for dim=1:dimension
            points(((c-1)*n_points+1):c*n_points, dim) = normrnd(means(c,dim), point_sigma, [n_points, 1]);
        end
    end
end

% Function for adding up mass in correct alignment
function alignment = eval_alignment(coupling, block_size, n_blocks)
    alignment = 0;
    for b=1:n_blocks
        for i=1:block_size
            idx = (b-1)*block_size + i;
            alignment = alignment + coupling(idx, b);
        end
    end
end
