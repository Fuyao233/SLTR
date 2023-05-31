function [] = Prox_precomputation_func(index_list)

for num = index_list
    addpath('UCF-101/')
addpath('tensor_toolbox/')
warning off
rng(42)

%load(sprintf('fMRI_data/P%d_res/P%d_invertX_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_X_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_Y_mat.mat', case_num, case_num))
% num = 3;
load(sprintf('UCF-101/data/Group%d/invertX.mat', num))
load(sprintf('UCF-101/data/Group%d/X.mat', num))
load(sprintf('UCF-101/data/Group%d/Y.mat', num))
load('UCF101_cv_epsilon.mat')
load('UCF101_group_epsilon_eig.mat')
mkdir(sprintf('UCF-101/res/Group%d',num))
format short g
X = tensor(X);
Y = tensor(double(Y));
invertX = tensor(invertX);
p = size(invertX);
M = length(p)-1;
N = p(end);

X = sptensor(X);
invertX = sptensor(invertX);

% split into training and testing set
splitPoint = floor(N*0.8);
indices = randperm(N);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);

if length(size(X)) == 5
    trainY = Y(trainIndices,:);
    trainX = X(trainIndices, :,:,:,:);
    trainInvertX = invertX(:,:,:,:, trainIndices);
    testY = Y(testIndices,:);
    testX = X(testIndices, :,:,:,:);
    testInvertX = invertX(:,:,:,:, testIndices);
else
    trainY = Y(trainIndices,:);
    trainX = X(trainIndices, :,:,:);
    trainInvertX = invertX(:,:,:, trainIndices);
    testY = Y(testIndices,:);
    testX = X(testIndices, :,:,:);
    testInvertX = invertX(:,:,:, testIndices);
end

N_train = size(trainY);
N_train = N_train(1);
N_test = size(testY);
N_test = N_test(1);

disp('===== Prox_Remurs =====')
    rho = 0.8; % learning rate
    minDiff=1e-4;
    maxIter=1000;
    matrixY=reshape(trainY.data,[splitPoint 1]);
    p = size(X);
    p = p(2:end);

    % calculate the startAppro
    % startApprox = Prox_Remurs_getStartApprox(X)

    epsilon_list = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0, -1]; % '-1' represent that the epsilon is adopted as the minimization of the eig
    % epsilon_list = [-1]; % debug 

    % CV
    disp('Start cv......')
    % disp(size(invertX))
    [bestPair, Res] = cv_Prox_Remurs_epsilon(double(invertX), Y, p,...
    [10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0],...
    [10^-4, 5*10^-4, 10^-3, 5*10^-3],...
    epsilon_list,...
    rho, 5, maxIter, minDiff,cv_epsilon(num+1,:));
    disp('Finish cv!')
    % bestPair{1} = [0.1, 0.2];
    
    save(sprintf('Prox_precomputation_res/Group%d.mat', num), 'Res')
    for idx_e = 1:length(epsilon_list)  
        cvTau = bestPair{idx_e}(1);
        cvLambda = bestPair{idx_e}(2);
        cvEpsilon = epsilon_list(idx_e);
        if cvEpsilon < 0
            cvEpsilon = group_epsilon_eig{num+1}(1); % 整个训练集的最大特征值
        end

        tic
        [estimatedW, errSeq, operator_group] = Prox_Remurs(double(trainInvertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
        t = toc;
        fprintf('Elapsed time is %.4f sec\n', t)

        % ROC and AUC
        predY = ttt(tensor(testX), tensor(estimatedW), 2:M+1, 1:M); 
        predY = reshape(double(predY),[N-splitPoint 1]);
        disp(size(tensor(predY)))
        disp(size(tensor(testY)))
        [rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);

        correct_cnt = 0;
        for i=1:length(predY)
            temp_pred = 0;
            if predY(i) > 0.5
                temp_pred = 1;
            end
            if temp_pred == testY(i)
                correct_cnt = correct_cnt + 1;
            end
        end
        fprintf('The correct rate of Prox_Remurs is %f\n', correct_cnt/length(predY));

        fprintf('The AUC value of Prox_Remurs is %f\n', AUC);
        % save the ROC curve
        Prox_ROC.X = rocX;
        Prox_ROC.Y = rocY;
        Prox_ROC.T = rocT;
        Prox_ROC.AUC = AUC;
        Prox_ROC.true_Y = testY;
        Prox_ROC.pred_Y = predY;
        Prox_ROC.t = t;
        Prox_ROC.bet_par = [cvTau, cvLambda, cvEpsilon];
        Prox_ROC.estimatedW = estimatedW;
        % Prox_ROC.cv_time = cv_time;
        save(sprintf('UCF-101/Prox_precomputation_res/Prox_ROC_g%d_epsilon%d.mat', num, idx_e), 'Prox_ROC')
    end
end
end

