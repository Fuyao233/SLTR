%% Experiment Setups
%{
pList  = [...
    [100 100 5];...
    [110 110 5];...
    [120 120 5];...
    [130 130 5];...
    [140 140 5];...
    [150 150 5];...
    ];
N = 2000;
%}

% pList  = [...
%     [10 10 5];...
%     [15 15 5];...
%     [20 20 5];...
%     [25 25 5];...
%     [30 30 5];...
%     [35 35 5];...
%     [40 40 5];...
%     ];
% N = [40 90 160 250 360 490 640]; %0.08

pList = [...
    [30 30 5 0];...
    [35 35 5 0];...
    [40 40 5 0];...
    [20 20 10 5];...
    [25 25 10 5];...
    [30 30 10 5];...
];
N = [360, 490, 640, 1600, 2500, 3600];
%pList  = [[10 10 5];];
%N = [50];

addpath('tensor_toolbox/');
addpath('4D_test/');
warning off;

% Prox_Remur_res = [];
% Remurs_res = [];
% SURF_res = [];
% Ela_res = [];
% Lasso_res = [];

rng(114514);
for dim_idx = 1:length(pList)
dim_idx = 4;
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;


% parameters for generating datasets��
% options.p = [20 20 20];
options.p = pList(dim_idx,:);
if options.p(end) == 0
   options.p = options.p(1:end-1);
end
%options.N = 100;
options.N = N(dim_idx);
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
if length(options.p) == 3
    [X, W, Y, Xvec, Wvec, invertX] = sparseGenerate(options);
else
    [X, W, Y, Xvec, Wvec, invertX] = sparse4DGenerate(options);
end
disp(options)

%% Experiment settings
repeat =2;

%% Remurs

% parameter settings
addpath('RemursCode/Code/')
disp('===== Remurs =====')
setting = expSet;
epsilon=1e-4;
iter=1000;
% time cost
totalTime = 0;
totalMSE = 0;
totalEE = 0;

matrixY=reshape(Y.data,[options.N 1]);

[cvAlpha, cvBeta, cv_time] = cv_Remurs(double(invertX), matrixY, options.p,...
        [5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0],...
        [5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0],...
        1, iter, epsilon);
% cvAlpha = 5;
% cvBeta = 1;

for it = 1:repeat
    tic
    [estimatedW, errList] = Remurs(double(invertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
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
% fprintf('cv_time is %.4f\n',cv_time)


% Remur.totalTime = totalTime / repeat;
% Remur.totalMSE = totalMSE / repeat;
% Remur.totalEE = totalEE / repeat;
% Remur.cv_time = cv_time;
% if dim_idx == 1
%     Remurs_res = repmat(Remur, 1, length(N));
% end
% 
% Remurs_res(dim_idx) = Remur;
% save('simulation_res/3D&4D/Remurs_res','Remurs_res')

% break

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
iteration_time = 0;
matrixY=reshape(Y.data,[options.N 1]);

[cvTau, cvLambda, cvEpsilon, cv_time] = cv_Prox_Remurs(double(invertX), matrixY, options.p,...
        [10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0],...
        [10^-4, 5*10^-4, 10^-3, 5*10^-3],...
        [0.1, 0.2, 0.3, 0.4],...
    rho, 1, maxIter, minDiff);

% cvTau = 0.01;
% cvLambda = 1e-4;
% cvEpsilon = 0.1;

for it = 1:repeat
    tic
    [estimatedW, errSeq, it_time] = Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
    t = toc;
    totalTime = totalTime + t;
    iteration_time = iteration_time + it_time;
    predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
    totalMSE = totalMSE + (norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N);
    error = reshape(W.data, options.p)-estimatedW;
    totalEE = totalEE + (norm(tensor(error)) /norm(tensor(W)));
    %totalEE = totalEE + (norm(tensor(error)) / prod(options.p)); 
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
fprintf('Response  Error is %.4f\n',totalMSE / repeat)
fprintf('Estimation Error is %.4f\n',totalEE / repeat)

% Prox_Remur.totalTime = totalTime / repeat;
% Prox_Remur.totalMSE = totalMSE / repeat;
% Prox_Remur.totalEE = totalEE / repeat;
% Prox_Remur.part_time = iteration_time / repeat;
% Prox_Remur.cv_time = cv_time;
% 
% if dim_idx == 1
%     Prox_Remur_res = repmat(Prox_Remur, 1, length(N));
% end
% 
% Prox_Remur_res(dim_idx) = Prox_Remur;
% save('simulation_res/3D&4D/Prox_Remur_res','Prox_Remur_res')


%% Lasso
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Lasso =====')
LassoOpt.alpha = 1;
LassoOpt.nlambda = 100;
%LassoOpt.lambda_min = 0.05;
LassoOpt = glmnetSet(LassoOpt);
totalTime = 0;
% time cost
for it = 1:repeat
    tic
    fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
% find out the best lambda and errors correspondingly
allPredY = glmnetPredict(fit, Xvec(end-testIndex+1:end,:), [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(Y(end-testIndex+1:end))-predY)) / testIndex;
    if i == 1
        minError = response_errors(1);
        minIndex = 1;
    else
        if response_errors(i) < minError
            minError = response_errors(i);
            minIndex = i;
        end
    end
end
fprintf('Response Error is %.4f \n', response_errors(1,minIndex))
predW = fit.beta(1:end,minIndex);
fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/norm(tensor(Wvec)))
%fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/prod(options.p))

Lasso.totalTime = totalTime / repeat;
Lasso.totalMSE = response_errors(1,minIndex);
Lasso.totalEE = norm(tensor(predW-Wvec))/norm(tensor(Wvec));

disp(dim_idx)

if dim_idx == 1

    Lasso_res = repmat(Lasso, 1, length(N));
end

Lasso_res(dim_idx) = Lasso;
% save('simulation_res/3D&4D/Lasso_res','Lasso_res')


%% Elasticnet (vectorize X)
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Elasticnet =====')
LassoOpt.alpha = 0.5;
LassoOpt.nlambda = 100;
%LassoOpt.lambda_min = 0.05;
LassoOpt = glmnetSet(LassoOpt);
totalTime = 0;
% time cost
for it = 1:repeat
    tic
    fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
% find out the best lambda and errors correspondingly
allPredY = glmnetPredict(fit, Xvec(end-testIndex+1:end,:), [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(Y(end-testIndex+1:end))-predY)) / testIndex;
    if i == 1
        minError = response_errors(1);
        minIndex = 1;
    else
        if response_errors(i) < minError
            minError = response_errors(i);
            minIndex = i;
        end
    end
end
fprintf('Response Error is %.4f \n', response_errors(minIndex))
predW = fit.beta(1:end,minIndex);
%predW = predW';
fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/norm(tensor(Wvec)))
%fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/prod(options.p))

Ela.totalTime = totalTime / repeat;
Ela.totalMSE = response_errors(minIndex);
Ela.totalEE = norm(tensor(predW-Wvec))/norm(tensor(Wvec));

if dim_idx == 1
    Ela_res = repmat(Ela, 1, length(N));
end

Ela_res(dim_idx) = Ela;

% save('simulation_res/3D&4D/Ela_res','Ela_res')

%% SURF

% parameter settings
addpath('SURF_code/')
addpath('SURF_code/tensorlab/')
disp('===== SURF =====')
%epsilon = 0.1;
%xi = epsilon^2 / 2; % [Jiaqi Zhang] set to the value recomended in the paper
%alpha = 1;
absW = 1e-3;
[cvAlpha, cvEpsilon, cvR, cv_time] = cv_SURF(double(invertX), Xvec, double(Y), options.p,...
           [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
           [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
           [1 2],...
    1, absW);
% time cost
totalTime = 0;
tmpY = Y;
for it = 1:repeat
    estimatedW = zeros(cvR,prod(options.p));
    tic
    for r =1:cvR
        [W_r, residual] = MyTrain(double(invertX), Xvec, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
        tmpY = residual;
        estimatedW(r,:) = W_r;     
    end
    t = toc;
    totalTime = totalTime + t; 
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
% compute errors
estimatedWVec = zeros(1,prod(options.p)); 
for r = 1:cvR
    estimatedWVec = estimatedWVec + estimatedW(r,:);
end
error = Wvec'-estimatedWVec;
predY = zeros(options.N, 1);
%vecX = tenmat(invertX, 4);
vecX = tenmat(X, 1);
vecX = vecX.data;
for i = 1:options.N
    predY(i) = vecX(i,:) * estimatedWVec';
end
Y = tensor(Y.data, [options.N 1]);
Y = Y.data;
fprintf('Response Error is %.6f\n',norm(tensor(predY - Y)) / options.N)
fprintf('Estimation Error is %.6f\n',norm(tensor(error)) / norm(tensor(Wvec)))

SURF.totalTime = totalTime / repeat;
SURF.totalMSE = norm(tensor(predY - Y)) / options.N;
SURF.totalEE = norm(tensor(error)) / norm(tensor(Wvec));
SURF.cv_time = cv_time;

if dim_idx == 1
    SURF_res = repmat(SURF, 1, length(N));
end

SURF_res(dim_idx) = SURF;

% save('simulation_res/3D&4D/SURF_res','SURF_res')

end