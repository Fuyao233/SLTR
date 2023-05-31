clc,clear;

%% Experiment Setting
[setting] = expSet;

addpath(genpath('tensor_toolbox_2.1'));
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
    %% Train--test splitting
    indexTest  = indexTotal((testFold-1)*numUnit+1 : (testFold-1)*numUnit + setting.numTest);
    indexTrain = setdiff(indexTotal,indexTest);
    
    %%  Read train and test data
    TrainData_3D = Data3D(:,:,:,indexTrain);
    TestData_3D  = Data3D(:,:,:,indexTest);

    %% Read train and test labels
    yTrain = label(indexTrain');
    yTest  = label(indexTest');
    
    %% Select parameters via cross validation
    %best = cross_validation_Remurs(TrainData_3D, yTrain, setting, setting.epsilon);
    
    %% Train the model (by default the maximum iteration is 1000)
    N = 100;
    p = [10 10 10];
    X = tenrand([p N]);
    W = tenrand(p);
    Y = ttt(X,W,1:3,1:3);
    err = 0.1*tenrand([N 1]);
    err = tensor(err.data, N);
    Y = Y + err;
    Y_mean = mean(Y.data);
    for i =1:N
        if Y(i)>Y_mean
            Y(i)=1;
        else
            Y(i)=-1;
        end
    end
    Y=reshape(Y.data,[N 1]);
    X=reshape(X.data,[p N]);
    tic
    %[tW, errList] = Remurs(TrainData_3D, yTrain, best.alpha, best.beta, setting.epsilon, 10);
    [tW, errList] = Remurs(X, Y, 0.1, 0.1,1e-4, 10);
    t=toc;
    fprintf('Time cost is %.4f',t)
    
    %{
    %% Predict
    X_test = reshape(TestData_3D, [], size(TestData_3D, 4));
    X_test = X_test';
    
    y_predict = X_test * tW(:);
    threshold = (max(label) + min(label))/2;
    y_predict(find(y_predict >= threshold)) = max(label);
    y_predict(find(y_predict <  threshold)) = min(label);
    
    %% Compute the accuracy, number of nonzero coefficients and best alpha, beta for each fold 
    fold.acc(testFold)   = length(find(y_predict == yTest)) / length(yTest);
    fold.num(testFold)   = length(find(tW(:) ~= 0));
    fold.alpha(testFold) = best. alpha;
    fold.beta(testFold)  = best.beta;
    
    save([ResultFolder '/Remurs_P',num2str(subject),'_F',num2str(testFold)], 'tW','best', 'subject','testFold', 'fold');
    %}
end
%{
%% Compute mean and std of all the metrics.
result.avgAcc = mean(fold.acc);
result.avgNum = mean(fold.num);
result.stdAcc = std(fold.acc);
result.stdNum = std(fold.num);
result.avgAlpha = mean(fold.alpha);
result.avgBeta  = mean(fold.beta);
result.stdAlpha = std(fold.alpha);
result.stdBeta  = std(fold.beta);


save([ResultFolder '/Remurs_P',num2str(subject),'_Results'], 'result');

fprintf('Algorithm finished.\n');
%}





