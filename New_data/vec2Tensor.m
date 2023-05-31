function [res] = vec2Tensor(vecData, col2Coord, tenDims)
    addpath('../tensor_toolbox/')
%     res = tenzeros(tenDims);
%     for index = 1:length(vecData)
%         coord = double(col2Coord(index,:,:,:));
%         t = vecData(index);
%         res(coord) = t;
%     end
    res = sptensor(double(col2Coord),vecData,tenDims);