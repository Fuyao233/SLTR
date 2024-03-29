%{
Description:
    Fold a matrix into a tensor.

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}

function [X] = Fold(X, dim, i)
dim = circshift(dim, [1-i, 1-i]);
X = shiftdim(reshape(X, dim), length(dim)+1-i);