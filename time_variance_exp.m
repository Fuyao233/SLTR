sparList = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
noiseList = [0.1];

for i=1:length(sparList)
    for j=1:length(noiseList)
        pars{i,j}=[sparList(i), noiseList(j)];
    end
end
%% Experiment Setups
warning off;
repeat = 10;
times = zeros([5 repeat]);
timeSet = ['Prox', 'Remur', 'Lasso', 'ENet', 'SURF'];

for i = 1:(length(sparList)*length(noiseList))
    for it=1:repeat
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;
addpath('tensor_toolbox/');
% parameters for generating datasets¡£
options.p = [20 20 20];
%options.p = [30 30 5];
options.N = 160;
options.R = 1;
par=pars{i};
options.sparsity = par(1); 
options.noise_coeff = par(2);
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
fprintf('%d-th iteration\n', it)

%% Remurs

% parameter settings
addpath('RemursCode/Code/')
disp('===== Remurs =====')
setting = expSet;
epsilon=1e-4;
iter=1000;
% time cost
matrixY=reshape(Y.data,[options.N 1]);

%[cvAlpha, cvBeta] = cv_Remurs(double(invertX), matrixY, options.p,...
%    [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
%    [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
%    5, iter, epsilon);

cvAlpha = 0.01;
cvBeta = 0.01;
tic
[estimatedW, errList] = Remurs(double(invertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
t = toc;
times(2,it)=t;
%% Prox_Remurs

% parameter settings
disp('===== Prox_Remurs =====')
cvTau = 0.5;
cvLambda = 5e-3;
cvEpsilon = 0.8;
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
% time cost
matrixY=reshape(Y.data,[options.N 1]);

%[cvTau, cvLambda, cvEpsilon] = cv_Prox_Remurs(double(invertX), matrixY, options.p,...
%    [0, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1],...
%    [0, 10^-4, 5*10^-4, 10^-3, 5*10^-3, 10^-2, 5*10^-2],...
%    [0, 0.1, 0.2, 0.3, 0.4, 0.5],...
%    rho, 5, maxIter, minDiff);

tic
[estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
t = toc;
times(1,it)=t;
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
% time cost
tic
fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
t = toc;
times(3,it)=t;
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
% time cost

tic
fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
t = toc;
times(4,it)=t;

%% SURF
% parameter settings
addpath('SURF_code/')
addpath('SURF_code/tensorlab/')
disp('===== SURF =====')
cvEpsilon = 0.1;
%xi = epsilon^2 / 2; % [Jiaqi Zhang] set to the value recomended in the paper
cvAlpha = 1;
absW = 1e-3;
cvR = 5;
%[cvAlpha, cvEpsilon, cvR] = cv_SURF(double(invertX), Xvec, double(Y), options.p,...
%    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1 5e-1 1],...
%    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1 5e-1 1],...
%    [2 4 6 8 10],...
%    5, absW);
% time cost
totalTime = 0;
tmpY = Y;
estimatedW = zeros(cvR,prod(options.p));
tic
for r =1:cvR
    [W_r, residual] = MyTrain(double(invertX), Xvec, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
    tmpY = residual;
    estimatedW(r,:) = W_r;     
end
t = toc;
times(5,it)=t;

    end
fprintf('==================== %f sparsity ===================\n',par(1))    


disp('Prox_Remurs')
fprintf('Average: %.4f; Std: %.4f\n', mean(times(1,:)), std(times(1,:)))
disp('Remurs')
fprintf('Average: %.4f; Std: %.4f\n', mean(times(2,:)), std(times(2,:)))
disp('Lasso')
fprintf('Average: %.4f; Std: %.4f\n', mean(times(3,:)), std(times(3,:)))
disp('ENet')
fprintf('Average: %.4f; Std: %.4f\n', mean(times(4,:)), std(times(4,:)))
disp('SURF')
fprintf('Average: %.4f; Std: %.4f\n', mean(times(5,:)), std(times(5,:)))
end


    