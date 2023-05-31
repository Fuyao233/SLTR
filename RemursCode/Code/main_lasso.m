clc,clear;

%% Experiment Setting
[setting]=expSet;

addpath(genpath('SLEP_package_4.1'));

ResultFolder = 'Results';

mkdir(ResultFolder);


dataset = 'CMU';
filename = 'data-science-P';
datapath = ['../Data/' dataset '/'];

%% We are using re-scaled (0.3 of each dimension) the original 3D data of subject 1 to show how to use the algorithm)
subject = 1;
Data = load([datapath 'RescaledP1.mat']);

%% Read label and the data
label = Data.label;
Data3D = Data.data;

nVoxels = size(Data3D,1)*size(Data3D,2)*size(Data3D,3);
setting.nVoxels = nVoxels;

indexTotal = 1:length(label);

numFold = setting.testFold;
numUnit  = length(indexTotal) / numFold;


for testFold = 1:numFold
    fprintf('testFold: %d\n\n', testFold)
    %% Train--test
    indexTest  = indexTotal((testFold-1)*numUnit+1 : (testFold-1)*numUnit + setting.numTest);
    indexTrain = setdiff(indexTotal,indexTest);
    
    %% Read train and test data
    TrainData_3D = Data3D(:,:,:,indexTrain);
    TestData_3D  = Data3D(:,:,:,indexTest);
    
    TrainData_1D = reshape(TrainData_3D, size(Data3D,1)*size(Data3D,2)*size(Data3D,3), size(TrainData_3D,4));
    TestData_1D  = reshape(TestData_3D,  size(Data3D,1)*size(Data3D,2)*size(Data3D,3), size(TestData_3D, 4));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    yTrain = label(indexTrain');
    yTest  = label(indexTest');
    
    %% Select parameters via cross validation
    best = cross_validation_lasso(TrainData_1D', yTrain, setting);
    
    %% Train the model
    [w, funVal] = LeastR(TrainData_1D', yTrain, best.beta, []);   
    
    %% Predict
    X_test = TestData_1D';
    
    y_predict = X_test * w;
    threshold = (max(label)+min(label))/2;
    y_predict(find(y_predict >= threshold)) = max(label);
    y_predict(find(y_predict <  threshold)) = min(label);
    
    %% Compute the accuracy, number of nonzero coefficients and best alpha, beta for each fold 
    fold.acc(testFold)   = length(find(y_predict == yTest)) / length(yTest);
    fold.num(testFold)   = length(find(w ~= 0));
    fold.beta(testFold)  = best.beta;
    
    save([ResultFolder '/Lasso_P',num2str(subject),'_F',num2str(testFold)], 'w', 'fold');
end
%% Compute mean and std of all the metrics.
result.avgAcc = mean(fold.acc);
result.avgNum = mean(fold.num);
result.stdAcc = std(fold.acc);
result.stdNum = std(fold.num);
result.avgBeta  = mean(fold.beta);
result.stdBeta  = std(fold.beta);

save([ResultFolder '/Lasso_P',num2str(subject),'_Results'], 'result');

fprintf('Algorithm finished.\n');





