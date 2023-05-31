function [] = getGroupEpsilonEig(head, tail)

for num = (head: tail)

addpath('UCF-101/')
addpath('tensor_toolbox/')
warning off

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

    X = double(trainInvertX);
    p = size(X);
    prodP = prod(p(1:end-1));
    N = p(end);
    dims = ndims(X);
    M = dims - 1;
    vecX = Unfold(X, p, dims);
    disp(num);
tic
    eps =  abs(max(eig(vecX' * vecX)));
time = toc;
    group_epsilon_eig{num+1} = [eps, time];
    save('UCF101_group_epsilon_eig.mat', 'group_epsilon_eig')

end

end

