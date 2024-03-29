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
    vecX = Unfold(X, p, dims);
    % Original approximation: (X^TX + epsilon I)^{-1} * X^Ty
    disp('Pre-computation')
    if epsilon > 0
        tic;
        [L, U] = factor(vecX, 1);
        numFea   = size(vecX,2);
        numSam   = size(vecX,1);
        q = vecX' * Y;
        if numSam >= numFea
            startApprox = U \ (L \ q);
        else
            startApprox = q - vecX' * ( U \ (L \ vecX * q));
        end
        
    else
        tic;
        [L, U] = factor(vecX, 1);
        semi_pos = L * U;
        epsilon = abs(max(eig(semi_pos)));
        disp('epsilon');
        disp(epsilon);
        
        numFea   = size(vecX,2);
        numSam   = size(vecX,1);
        q = vecX' * Y;
        if numSam >= numFea
            startApprox = U \ (L \ q);
        else
            startApprox = q - vecX' * ( U \ (L \ vecX * q));
        end
    end
    startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
    startApprox(isnan(startApprox)) = 0;
    totalTime = toc;

    disp('Time for pre-computation:')
    disp(totalTime)