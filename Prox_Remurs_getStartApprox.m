function [startApprox, totalTime] = Prox_Remurs_getStartApprox(X, Y, epsilon)
    if nargin < 3
       epsilon = 0.4; 
    end
    % epsilon = 0.4;
    p = size(X);
    prodP = prod(p(1:end-1));
    N = p(end);
    dims = ndims(X);
    M = dims - 1;
    % vectorizing the 3D samples 
    % TODO: take matrix X as an input argument
    vecX = Unfold(X, p, dims);
    % Original approximation: (X^TX + epsilon I)^{-1} * X^Ty
    disp('Pre-computation')
    if epsilon > 0
        tic;
        startApprox = ( (vecX' * vecX + epsilon * eye(prodP)) \ eye(prodP) ) * vecX' * Y;
    else
        tic;
        semi_pos = vecX' * vecX;
        epsilon = abs(max(eig(semi_pos)));
        disp('epsilon');
        disp(epsilon);
        startApprox = ( (semi_pos + epsilon * eye(prodP)) \ eye(prodP) ) * vecX' * Y;
    end
    startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
    startApprox(isnan(startApprox)) = 0;
    totalTime = toc;

    disp('Time for pre-computation:')
    disp(totalTime)