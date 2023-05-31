clc,clear;

%% Experiment Setting
[setting]=expSet;

addpath('tensor_toolbox_2.1');
addpath(genpath('SLEP_package_4.1'));

%% Create a result folder if not existed
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

    %% Read train and test labels
    yTrain = label(indexTrain');
    yTest  = label(indexTest');
    
    %% Select parameters via cross validation
    best = cross_validation_lowRank(TrainData_3D, yTrain, setting);
    
    %% Train the model
    [tW, errList] = Remurs(TrainData_3D, yTrain, best.alpha, 0, setting.epsilon, 1000);
    
    %% Predict
    X_test = reshape(TestData_3D, [], size(TestData_3D, 4));
    X_test = X_test';
    
    y_predict = X_test * tW(:);
    threshold = (max(label) + min(label))/2;
    y_predict(find(y_predict >= threshold)) = max(label);
    y_predict(find(y_predict <  threshold)) = min(label);
    
    acc(testFold) = length(find(y_predict == yTest)) / length(yTest);
    num(testFold) = length(find(tW(:) ~= 0));
    
    %%
    fold.acc(testFold)   = length(find(y_predict == yTest)) / length(yTest);
    fold.num(testFold)   = length(find(tW(:) ~= 0));
    fold.alpha(testFold) = best.alpha;
    
    save([ResultFolder '/Remurs_LowRank_P',num2str(subject),'_F',num2str(testFold)], 'tW','best', 'subject','testFold', 'fold');
end
result.avgAcc = mean(fold.acc);
result.avgNum = mean(fold.num);
result.stdAcc = std(fold.acc);
result.stdNum = std(fold.num);
result.avgAlpha = mean(fold.alpha);
result.stdAlpha = std(fold.alpha);

save([ResultFolder '/Remurs_LowRank_P',num2str(subject),'_Results'], 'result');

fprintf('Algorithm finished.\n');






