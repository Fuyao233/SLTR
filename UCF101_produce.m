group_list = [1:10];
addpath('UCF-101/')
addpath('tensor_toolbox/')
warning off

for g = group_list
disp(g);
%load(sprintf('fMRI_data/P%d_res/P%d_invertX_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_X_mat.mat', case_num, case_num))
%load(sprintf('fMRI_data/P%d_res/P%d_Y_mat.mat', case_num, case_num))
load(sprintf('UCF-101/data/Group%d/invertX.mat', g))
load(sprintf('UCF-101/data/Group%d/X.mat', g))
load(sprintf('UCF-101/data/Group%d/Y.mat', g))
format short g
X = tensor(X);
Y = tensor(double(Y));
invertX = tensor(invertX);
p = size(invertX);
M = length(p)-1;
N = p(end);

X = sptensor(X);
invertX = sptensor(invertX);

X = double(invertX);
p = size(X);
prodP = prod(p(1:end-1));
N = p(end);
dims = ndims(X);
M = dims - 1;
vecX = Unfold(X, p, dims);
res = vecX' * vecX;

save(sprintf('UCF101_data_XX/group%d', g), 'res');


end
