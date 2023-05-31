function [] = getEpsilon(head, tail)
% calculate the cv_epsilon(maximum of eigenvalue) from Group_start to Group_end
%   for each group, there are five epsilon for 5 fold

for num = (head: tail)

addpath('UCF-101/')
addpath('tensor_toolbox/')
warning off

%load(sprintf('fMRI_data/P%d_res/P%d_invertX_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_X_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_Y_mat.mat', case_num, case_num))
% num = 0;
load(sprintf('UCF-101/data/Group%d/invertX.mat', num))
load(sprintf('UCF-101/data/Group%d/X.mat', num))
load(sprintf('UCF-101/data/Group%d/Y.mat', num))
mkdir(sprintf('UCF-101/res/Group%d',num))
format short g
X = tensor(X);
Y = tensor(double(Y));
invertX = tensor(invertX);
p = size(invertX);

N = p(end);

X = sptensor(X);
invertX = sptensor(invertX);

rng(42)
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

disp('===== Prox_Remurs =====')
    
    X = double(invertX);
    
    N = size(X);
    N = N(end);
    fold = 5;
    cvp = cvpartition(N, 'Kfold', fold);

    for f = 1:cvp.NumTestSets
        disp('cvp.NumTestSets:');
        disp(f);
        trains = cvp.training(f);
        tests =  cvp.test(f);
        trainIndex = [];
        testIndex = [];
        for i = 1:N
            if trains(i) == 1
                trainIndex = [trainIndex i];
            end
            if tests(i) == 1
                testIndex = [testIndex i];
            end
        end
        
        if length(size(X)) == 5
            % 5D
            Xtrain = X(:,:,:,:,trainIndex);
            Ytrain = Y(trainIndex,:);
        end

        if length(size(X)) == 4
            % 4D
            Xtrain = X(:,:,:,trainIndex);
            Ytrain = Y(trainIndex,:);
        end

        X = double(Xtrain);
        Y = double(Ytrain);
        p = size(X);
        N = p(end);
        dims = ndims(X);
        vecX = Unfold(X, p, dims);
        tic;
        epsilon = abs(max(eig(vecX' * vecX)));
        epsilon_time = toc;
        
        disp([num, f, epsilon]); 
        cv_epsilon{num+1, f} = [epsilon, epsilon_time];
    end
    save('UCF101_cv_epsilon.mat', 'cv_epsilon')

end



end

