addpath('../tensor_toolbox/')
%% Load data 
case_num = 9
load(sprintf('data-science-P%d.mat', case_num));
tenDims = size(meta.coordToCol);
col2Coord = meta.colToCoord;
N = meta.ntrials;

%% Construct the Dataset
toolIndex = [];
animalIndex = [];
for i = 1:N
    if strcmp(info(i).cond, 'tool') || strcmp(info(i).cond, 'furniture')
        toolIndex = [toolIndex, i];
    end
    if strcmp(info(i).cond, 'animal') || strcmp(info(i).cond, 'insect')
        animalIndex = [animalIndex, i];
    end
end

X = tenzeros([length(animalIndex)+length(toolIndex), tenDims]);
invertX = tenzeros([tenDims, length(animalIndex)+length(toolIndex)]);
Y = tenzeros([length(animalIndex)+length(toolIndex), 1]);
% animals (with lable 1)
for i=1:length(animalIndex)
    Y(i) = 1;
    X(i,:,:,:) = vec2Tensor(data{animalIndex(i),1},col2Coord, tenDims);
    invertX(:,:,:,i) = X(i,:,:,:);
end
% tool (with lable 0)
offset = length(animalIndex);
for i=1:length(toolIndex)
    Y(i+offset) = 0;
    X(i+offset,:,:,:) = vec2Tensor(data{toolIndex(i),1},col2Coord, tenDims);
    invertX(:,:,:,i+offset) = X(i+offset,:,:,:);
end

% save(sprintf('P%d_invertX_mat.mat', case_num), "invertX")
% save(sprintf('P%d_X_mat.mat', case_num), "X")
% save(sprintf('P%d_Y_mat.mat', case_num), "Y")