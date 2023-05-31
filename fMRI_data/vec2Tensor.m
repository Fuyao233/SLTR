function [res] = vec2Tensor(vecData, col2Coord, tenDims)
    addpath('../tensor_toolbox/')
    res = tenzeros(tenDims);
    for index = 1:length(vecData)
        coord = col2Coord(index,:,:,:);
        res(coord) = vecData(index);
    end