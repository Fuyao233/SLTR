%% TODO: split 120 samples into a training set and a test set
addpath('fMRI_data')
addpath('tensor_toolbox/')

for case_num=1:6
%case_num = 1;
load(sprintf('fMRI_data/P%d_invertX_mat.mat', case_num))
load(sprintf('fMRI_data/P%d_X_mat.mat', case_num))
load(sprintf('fMRI_data/P%d_Y_mat.mat', case_num))
format short g
p = size(invertX);
M = length(p)-1;
N = p(end);

% split into training and testing set
splitPoint = floor(N*0.8);
indices = randperm(N);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
trainY = Y(trainIndices,:);
trainX = X(trainIndices, :,:,:);
trainInvertX = invertX(:,:,:, trainIndices);
testY = Y(testIndices,:);
testX = X(testIndices, :,:,:);
testInvertX = invertX(:,:,:, testIndices);

%% Prox_Remurs (0, 0, 0.25)
% parameter settings
disp('===== Prox_Remurs =====')
fprintf('Project num is %d \n',case_num)
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
matrixY=reshape(trainY.data,[splitPoint 1]);
cvTau = 1e-5;
cvLambda = 1e-5;
cvEpsilon = 0.25;
%[cvTau, cvLambda, cvEpsilon] = cv_Prox_Remurs(double(trainInvertX), matrixY, p,...
%    [0, 10^-2, 10^-1, 10^0, 10^1],...
%    [0, 10^-4, 10^-3, 10^-2, 10^-1],...
%    [0, 0.1, 0.2, 0.3, 0.4, 0.5],...
%    rho, 5, maxIter, minDiff);
fprintf('cvTau is %f \n',cvTau)
fprintf('cvLambda is %f \n',cvLambda)
fprintf('cvEpsilon is %f \n',cvEpsilon)
% main step
tic
[estimatedW, errSeq] = Prox_Remurs(double(trainInvertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% ROC and AUC
predY = ttt(tensor(testX), tensor(estimatedW), 2:M+1, 1:M); 
predY = reshape(double(predY),[N-splitPoint 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of Prox_Remurs is %f\n', AUC);

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

% save the ROC curve
Prox_ROC.X = rocX;
Prox_ROC.Y = rocY;
Prox_ROC.T = rocT;
Prox_ROC.AUC = AUC;
Prox_ROC.true_Y = testY;
Prox_ROC.pred_Y = predY;
save(sprintf('fMRI_data/P%d_res/Prox_ROC.mat', case_num), 'Prox_ROC')
end