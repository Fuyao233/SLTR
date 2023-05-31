addpath('UCF-101/')
addpath('tensor_toolbox/')
warning off

%load(sprintf('fMRI_data/P%d_res/P%d_invertX_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_X_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_Y_mat.mat', case_num, case_num))
load('UCF-101/invertX_RGB_spared80_g7.mat')
load('UCF-101/X_RGB_spared80_g7.mat')
load('UCF-101/Y_RGB_spared80_g7.mat')
format short g
X = tensor(X);
Y = tensor(double(Y));
invertX = tensor(invertX);
p = size(invertX);
M = length(p)-1;
N = p(end);

% 本地调试，减小数据规模
% X = X(:,1:2,:,:,:);
% invertX = invertX(1:2,:,:,:,:);

X = sptensor(X);
invertX = sptensor(invertX);
% disp(size(X))
% split into training and testing set
splitPoint = floor(N*0.8);
indices = randperm(N);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
trainY = Y(trainIndices,:);
trainX = X(trainIndices, :,:,:,:);
trainInvertX = invertX(:,:,:,:, trainIndices);
testY = Y(testIndices,:);
testX = X(testIndices, :,:,:,:);
testInvertX = invertX(:,:,:,:, testIndices);

N_train = size(trainY);
N_train = N_train(1);
N_test = size(testY);
N_test = N_test(1);
% disp('Y');
disp(sum(double(Y))/N);
% disp('trainY');
disp(sum(double(trainY))/N_train);
% disp('testY');
disp(sum(double(testY))/N_test);

%% Prox_Remurs (0, 0, 0.25)
    % parameter settings
    disp('===== Prox_Remurs =====')
    rho = 0.8; % learning rate
    minDiff=1e-4;
    maxIter=1000;
    matrixY=reshape(trainY.data,[splitPoint 1]);
    p = size(X);
    p = p(2:end);

    % calculate the startAppro
    % startApprox = Prox_Remurs_getStartApprox(X)

    % CV
%     disp('Start cv......')
%     disp(size(invertX))
%     [cvTau, cvLambda, cvEpsilon, cv_time] = cv_Prox_Remurs(double(invertX), Y, p,...
%     [10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0],...
%     [10^-4, 5*10^-4, 10^-3, 5*10^-3],...
%     [0.1, 0.2, 0.3, 0.4],...
%     rho, 5, maxIter, minDiff);
%     disp('Finish cv!')
    cvTau = 0.01;
    cvLambda = 0.0001;
    cvEpsilon = 0.1;

    disp('size of trainInvertX')
    disp(size(trainInvertX))
    % main step
    fprintf('Class of invertX is %s \n',class(trainInvertX))
    tic
    [estimatedW, errSeq, operator_group] = Prox_Remurs(double(trainInvertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
    t = toc;
    fprintf('Elapsed time is %.4f sec\n', t)
    % fprintf('Before time is %.4f sec\n',operator_group(1))
    % fprintf('l1_operator time is %.4f sec\n',operator_group(2))
    % fprintf('Inf_operator time is %.4f sec\n',operator_group(3))
    % fprintf('Nuclear time is %.4f sec\n',operator_group(4))
    % fprintf('Spe_oprator time is %.4f sec\n',operator_group(5))
    % fprintf('After time is %.4f sec\n',operator_group(6)+operator_group(7))

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
    Prox_ROC.cv_time = cv_time;
    save(sprintf('UCF-101/res/Prox_ROC_RGB_g7.mat'), 'Prox_ROC')


% %% Remurs
% 
%     % parameter settings
%     addpath('RemursCode/Code/')
%     disp('===== Remurs =====')
%     disp('Start time')
% 
%     setting = expSet;
%     epsilon=1e-4;
%     iter=1000;
%     matrixY=reshape(trainY.data,[splitPoint 1]);
%     p = size(X);
%     p = p(2:end);
%     % cv
%     disp('Start cv......')
%     % disp(size(invertX))
%     [cvAlpha, cvBeta, cv_time] = cv_Remurs(double(invertX), Y, p,...
%     [5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0],...
%     [5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0],...
%     5, iter, epsilon);
%     disp('Finish cv!')
%     cvAlpha = 0.5;
%     cvBeta = 0.01;
%     epsilon = 1e-4;
%     % main step
% 
%     tic
%     [estimatedW, errList] = Remurs(double(trainInvertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
%     t = toc;
%     fprintf('Elapsed time is %.4f sec\n',t)
%     predY = ttt(tensor(testX), tensor(estimatedW), 2:M+1, 1:M);
%     predY = reshape(double(predY),[N-splitPoint 1]);
%     [rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
% 
%     correct_cnt = 0;
%     for i=1:length(predY)
%         temp_pred = 0;
%         if predY(i) > 0.5
%             temp_pred = 1;
%         end
%         if temp_pred == testY(i)
%             correct_cnt = correct_cnt + 1;
%         end
%     end
%     fprintf('The correct rate of Remurs is %f\n', correct_cnt/length(predY));
% 
%     fprintf('The AUC value of Remurs is %f\n', AUC);
%     % save the ROC curve
%     Remur_ROC.X = rocX;
%     Remur_ROC.Y = rocY;
%     Remur_ROC.T = rocT;
%     Remur_ROC.AUC = AUC;
%     Remur_ROC.true_Y = testY;
%     Remur_ROC.pred_Y = predY;
%     Remur_ROC.t = t;
%     Remur_ROC.estimatedW = estimatedW;
%     Remur_ROC.cv_time = cv_time;
%     save(sprintf('UCF-101/res/Remur_ROC_RGB_g7.mat'), 'Remur_ROC')
%     disp('Finish time')

%% SURF
    % parameter settings
    addpath('SURF_code/')
    addpath('SURF_code/tensorlab/')
    disp('===== SURF =====')
    % convet X to vector for Lasso and ENet
    vecX = tenmat(trainX, 1);
    vecX = vecX.data;

    absW = 1e-3;
    % cvAlpha = 0.1;
    % cvEpsilon = 0.08;
    % cvR = 5;
    disp('Start cv......')
    % clock
    [cvAlpha, cvEpsilon, cvR, cv_time] = cv_SURF(double(trainInvertX), vecX, double(trainY), p(1:M),...
       [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
       [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
       [1 2],...
       5, absW);
    disp('Finish cv!')
    % clock
    % time cost
    totalTime = 0;
    tmpY = trainY;
    % main procedure
    estimatedW = zeros(cvR,prod(p(1:M)));
    tic
    for r =1:cvR
        [W_r, residual] = MyTrain(double(trainInvertX), vecX, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
        tmpY = residual;
        estimatedW(r,:) = W_r;     
    end
    t = toc;
    fprintf('Elapsed time is %.4f sec\n',t)
    %clear invertX;
    % compute errors
    estimatedWVec = zeros(1,prod(p(1:M))); 
    for r = 1:cvR
        estimatedWVec = estimatedWVec + estimatedW(r,:);
    end
    predY = zeros(N-splitPoint, 1);
    vecX = tenmat(testX, 1);
    vecX = vecX.data;
    for i = 1:(N-splitPoint)
        predY(i) = vecX(i,:) * estimatedWVec';
    end
    testY = tensor(testY.data, [N-splitPoint 1]);
    testY = testY.data;
    [rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
    fprintf('The AUC value of SURF is %f\n', AUC);
    % save the ROC curve
    SURF_Roc.X = rocX;
    SURF_Roc.Y = rocY;
    SURF_Roc.T = rocT;
    SURF_Roc.AUC = AUC;
    SURF_Roc.true_Y = testY;
    SURF_Roc.pred_Y = predY;
    SURF_Roc.t = t;
    SURF_Roc.estimatedW = estimatedWVec;
    SURF_Roc.cv_time = cv_time;
    save(sprintf('UCF-101/res/SURF_ROC_RGB_g7.mat'), 'SURF_Roc')

% %% Split dataset only for LR
% % split into training and testing set
% splitPoint = floor(N*0.8);
% indices = randperm(N);
% trainIndices = indices(1:splitPoint);
% testIndices = indices(splitPoint+1:end);
% trainY = Y(trainIndices,:);
% trainX = X(trainIndices, :,:,:,:);
% %trainInvertX = invertX(:,:,:, trainIndices);
% testY = Y(testIndices,:);
% testX = X(testIndices, :,:,:,:);
% %testInvertX = invertX(:,:,:, testIndices);

% %     testY = Y;
% %     testX = X;

% %% Lasso
% % vectorize X
% clear invertX;
% % parameter settings
% addpath('GLMNET/');
% disp('===== Lasso =====')
% disp('Start time')
% LassoOpt.alpha = 1;
% LassoOpt.nlambda = 100;
% LassoOpt = glmnetSet(LassoOpt);
% % time cost
% % convet X to vector for Lasso and ENet
% vecX = tenmat(trainX, 1);
% vecX = vecX.data;
% tic
% fit = glmnet(vecX, double(trainY), [], LassoOpt);
% t = toc;
% fprintf('Elapsed time is %.4f sec\n',t)
% % find out the best lambda and errors correspondingly
% vecX = tenmat(testX, 1);
% vecX = vecX.data;
% allPredY = glmnetPredict(fit, vecX, [], 'response');
% lambdaNum = size(fit.lambda);
% lambdaNum = lambdaNum(1);
% response_errors = zeros(1, lambdaNum);
% for i = 1:lambdaNum
%     predY = allPredY(1:end,i);
%     response_errors(i) = norm(tensor(double(testY)-predY));
%     if i == 1
%         minError = response_errors(1);
%         minIndex = 1;
%     else
%         if response_errors(i) < minError
%             minError = response_errors(i);
%             minIndex = i;
%         end
%     end
% end
% predY = allPredY(1:end,minIndex);
% %predY = reshape(double(predY), [N-splitPoint 1]);
% predY = reshape(double(predY), [length(testIndices) 1]);
% LpredW = fit.beta(1:end,minIndex);
% [rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
% fprintf('The AUC value of Lasso is %f\n', AUC);
% % save the ROC curve
% Lasso_ROC.X = rocX;
% Lasso_ROC.Y = rocY;
% Lasso_ROC.T = rocT;
% Lasso_ROC.AUC = AUC;
% Lasso_ROC.true_Y = testY;
% Lasso_ROC.pred_Y = predY;
% Lasso_ROC.t =t;
% Lasso_ROC.estimatedW = LpredW;
% save(sprintf('UCF-101/res/Lasso_ROC_RGB_g7.mat'), 'Lasso_ROC')
% disp('Finish time')

% %% ENet
% % parameter settings
% addpath('GLMNET/');
% disp('===== Elasticnet =====')
% disp('Start time')
% LassoOpt.alpha = 0.5;
% LassoOpt.nlambda = 100;
% LassoOpt = glmnetSet(LassoOpt);
% % time cost
% % convet X to vector
% vecX = tenmat(trainX, 1);
% vecX = vecX.data;
% tic
% fit = glmnet(vecX, double(trainY), [], LassoOpt);
% t = toc;
% fprintf('Elapsed time is %.4f sec\n',t)
% % find out the best lambda and errors correspondingly
% vecX = tenmat(testX, 1);
% vecX = vecX.data;
% allPredY = glmnetPredict(fit, vecX, [], 'response');
% N = length(testY(:));
% lambdaNum = size(fit.lambda);
% lambdaNum = lambdaNum(1);
% response_errors = zeros(1, lambdaNum);
% for i = 1:lambdaNum
%     predY = allPredY(1:end,i);
%     response_errors(i) = norm(tensor(double(testY)-predY));
%     if i == 1
%         minError = response_errors(1);
%         minIndex = 1;
%     else
%         if response_errors(i) < minError
%             minError = response_errors(i);
%             minIndex = i;
%         end
%     end
% end
% predY = allPredY(1:end,minIndex);
% %predY = reshape(double(predY), [N-splitPoint 1]);
% predY = reshape(double(predY), [N 1]);
% EpredW = fit.beta(1:end,minIndex);
% [rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);

% % [prec, tpr, fpr, thresh] = prec_rec(double(testY), sigmoid(predY), 1);

% fprintf('The AUC value of Lasso is %f\n', AUC);
% % save the ROC curve
% ENet.X = rocX;
% ENet.Y = rocY;
% ENet.T = rocT;
% ENet.AUC = AUC;
% ENet.true_Y = testY;
% ENet.pred_Y = predY;
% ENet.t = t;
% ENet.estimatedW = EpredW;
% save(sprintf('UCF-101/res/ENet_ROC_RGB_g7.mat'), 'ENet')
% disp('Finish time')


