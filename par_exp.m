%% Experiment Setups
%{
pList  = [...
    [10 10 5];...
    [20 20 5];...
    [30 30 5];...
    [40 40 5];...
    [50 50 5];...
    [60 60 5];...
    [100 100 5];...
    [200 200 5];...
    [300 300 5];...
    [400 400 5];...
    ];
NList = [40, 160, 360, 640, 1000, 1440, 1000, 1000, 1000, 1000]; %0.08
%}
pList = [[30 30 5]];
NList = [3000];
warning off;

for i = 1:length(pList)
    
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;

addpath('tensor_toolbox/');
% parameters for generating datasets¡£
% options.p = [20 20 20];
options.p = pList(i,:);
%options.N = 100;
options.N = NList(i);
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
repeat =1;

%% Prox_Remurs
% parameter settings
disp('===== Prox_Remurs =====')
tau = 0.05;
lambda = 0.01;
epsilon = 0.1;
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
% time cost
totalTime = 0;
totalMSE = 0;
totalEE = 0;
matrixY=reshape(Y.data,[options.N 1]);
%{
[cvTau, cvLambda, cvEpsilon] = cv_Prox_Remurs(double(invertX), matrixY, options.p,...
    [0, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1],...
    [0, 10^-4, 5*10^-4, 10^-3, 5*10^-3, 10^-2, 5*10^-2],...
    [0, 0.1, 0.2, 0.3, 0.4, 0.5],...
    rho, 5, maxIter, minDiff);
%}
for it = 1:repeat
    tic
    [estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, tau, lambda, epsilon, rho, maxIter, minDiff);
    t = toc;
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

for it = 1:repeat
    tic
    [estimatedW] = Parallel_Prox_Remurs(double(invertX), matrixY, tau, lambda, epsilon, rho, maxIter, minDiff);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('[ Proximal Parallel ] Elapsed time is %f\n',totalTime/repeat)

%{
for it = 1:repeat
    tic
    [estimatedW, ~] = propar_Prox_Remurs(double(invertX), matrixY, tau, lambda, epsilon, rho, maxIter, minDiff);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('[ Proximal Parallel ] Elapsed time is %f\n',totalTime/repeat)


for it = 1:repeat
    tic
    [estimatedW] = modepar_Prox_Remurs(double(invertX), matrixY, tau, lambda, epsilon, rho, maxIter, minDiff);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('[ Mode Parallel ] Elapsed time is %f\n',totalTime/repeat)
%}
end