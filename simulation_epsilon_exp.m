pList  = [...
    [10 10 5];...
    [15 15 5];...
    [20 20 5];...
    [25 25 5];...
    [30 30 5];...
    [35 35 5];...
    [40 40 5];...
    ];
N = [40 90 160 250 360 490 640]; %0.08

%pList  = [[10 10 5];];
%N = [50];

rng = 42;
warning off;

for i = 1:length(pList)
    
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;

addpath('tensor_toolbox/');
% parameters for generating datasets。
% options.p = [20 20 20];
options.p = pList(i,:);
%options.N = 100;
options.N = N(i);
options.R = 5;
options.sparsity = 0.2; 
options.noise_coeff = 0.1;
M = length(options.p);
%{
    X -- a tensor with shape N x p1 x p2 x...x pM 
    W -- a tensor with shape p1 x p2 x...x pM 
    Y -- a trensor with shape N
    Xvec -- a matrix with shape N x (p1 * p2 *...* pM)
    Wvec -- a vector with shape (p1 * p2 *..* pM) x 1 
    invertX -- a tensor with shape p1 x p2 x...x pM x N
%}
%[X, W, Y, Xvec, Wvec, invertX] = generateData(options);
[X, W, Y, Xvec, Wvec, invertX] = sparseGenerate(options);
disp(options)

%% Experiment settings
repeat =2;

%% Prox_Remurs
% parameter settings
disp('===== Prox_Remurs =====')
%tau = 0.5;
%lambda = 5e-3;
%epsilon = 0.8;
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
% time cost
totalTime = 0;
totalMSE = 0;
totalEE = 0;
matrixY=reshape(Y.data,[options.N 1]);

% epsilon_list = [0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001];
% disp('start CV....');
% [bestPair, Res] = cv_Prox_Remurs_epsilon(double(invertX), matrixY, options.p,...
%     [10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0],...
%     [10^-4, 5*10^-4, 10^-3, 5*10^-3],...
%     epsilon_list,...
%     rho, 5, maxIter, minDiff);
% disp('finish CV !');

% Just for saving X^TX
bestPair{1} = [1, 1];
epsilon_list = [0.01];

% save(sprintf('Simulation_Prox_precomputation_res/Group%d.mat', i), 'Res')

    for idx_e = 1:length(epsilon_list)
        cvTau = bestPair{idx_e}(1);
        cvLambda = bestPair{idx_e}(2);
        cvEpsilon = epsilon_list(idx_e);
        % cvTau = 0.1;
        % cvLambda = 0.005;
        % cvEpsilon = 1;

        for it = 1:repeat
            tic
            [estimatedW, errSeq, ~, ~, mid] = Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
            t = toc;
            save(sprintf('Group%d.mat', i), 'mid');
            totalTime = totalTime + t;
            predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
            totalMSE = totalMSE + (norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N);
            error = reshape(W.data, options.p)-estimatedW;
            totalEE = totalEE + (norm(tensor(error)) /norm(tensor(W)));
            %totalEE = totalEE + (norm(tensor(error)) / prod(options.p)); 
        end

        fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
        fprintf('Response  Error is %.4f\n',totalMSE / repeat)
        fprintf('Estimation Error is %.4f\n',totalEE / repeat)

        Prox_res.totalTime = totalTime / repeat;
        Prox_res.totalMSE = totalMSE / repeat;
        Prox_res.totalEE = totalEE / repeat;

        % save(sprintf('Simulation_Prox_precomputation_res/Prox_ROC_g%d_epsilon%d.mat', i, idx_e), 'Prox_res')
    end
end