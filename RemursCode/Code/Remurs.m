function [tW, errList] = Remurs(tX, y, alpha, beta, epsilon, max_iter)

% Remurs: Regularized Multilinear Regression and Selection
%
% %[Prototype]%
% function [tW, errList, time]  = Remurs(tX, y, alpha, beta, epsilon, max_iter)
%
% %[Author Notes]%
% Author: Xiaonan Song, Wenwen Li, and Haiping Lu*                            
% *Email : hplu@ieee.org   or   eehplu@gmail.com  or h.lu@sheffield.ac.uk     
% Release date: 26th March 2019                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %[Algorithm]%:
% This function implements the regularized multilinear regression and selection
% (Remurs) algorithm presented in the follwing paper:
%   Song, Xiaonan and Lu, Haiping, 
%	"Multilinear Regression for Embedded Feature Selection with Application to fMRI Analysis", 
%	Thirty-First AAAI Conference on Artificial Intelligence (AAAI), pp. 2562-2568, 2017.
%   
% %[Remurs Model]%:
%
%   min 1/2 sum_{m=1}^M (y_m - <X, W>)^2 + alpha ||W||_{SNN} + beta ||W||_{1} 
%
%
% Please reference this paper when reporting work done using this code.
%
% %[Toolbox needed]%:
% This function needs the tensor toolbox available at 
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/
% This Remurs package includes tensor toolbox version 2.1 for convenience.
% This code also requires SLEP pakcage which are available at 
% https://github.com/jiayuzhou/SLEP
% This Remurs package include SLEP package version 4.1 for convenience.
%
% %[Syntax]%: [tW, errList, time]  = Remurs(tX, y, alpha, beta, epsilon, max_iter)
%
% %[Inputs]%:
%    tX: the input training data in tensorial representation, the last mode
%        is the sample mode. For Nth-order tensor data, tX is of 
%        (N+1)th-order with the (N+1)-mode to be the sample mode.
%        E.g., 30x20x10x100 for 100 samples of size 30x20x10
%
%    y: the ground truth class labels (1,-1) for the training data
%           E.g., a 100x1 vector if there are 100 samples
%
%    alpha: the hyper-parameter for the sum of tensor nuclear norm (SNN), this
%    can be tuned by cross-validation. Suggested range of this parameter is
%    in expSet.m
%
%    beta: the hyper-parameter for the L1 norm, this can be tuned by 
%    cross-validation. Suggested range of this parameter is in expSet.m
%    
%    epsilon: convergence stopping criteria, default as 1e-4
%    max_iter: maximum iteration stopping criteria, e.g., 1000
%
% %[Outputs]%:
%    tW: the coefficient tensor with N modes
%
%    errList: error at each iteration which is used for verifying if the
%    the convergence criteria is satisfied or not.
%
%    time: running time of Remurs
%
%
% %[Supported tensor order]%
% This function supports N=3, for other order N, please modify the codes accordingly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %[Notes]%:
% Orignally developed using Matlab R2015a. Current version developed using Matlab R2018a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set defalt parameters
if nargin < 4
    fprintf('Not enough parameters for Remurs! Terminate now.\n')
    return;
end

if nargin < 6
    max_iter = 1000;
end

if nargin < 5
    epsilon = 1e-4;
end

%% Initialize

totalTime = 0;

lambda   = 1;
rho      = 1/lambda;
N        = ndims(tX) - 1; % Number of modes.
size_tX  = size(tX);
dim      = size_tX(1:N);
X        = Unfold(tX, size(tX), N+1); % Data in vector form. Matrix: M*d.

% tic
Xty      = X'*y;
% totalTime = totalTime + toc;

numFea   = size(X,2);
numSam   = size(X,1);
tW        = zeros([dim,1]); % Tensor W
tU        = zeros([dim,1]); % Tensor U
tA        = zeros([dim,1]); % Tensor A
for n = 1:N
    tV{n} = zeros([dim,1]); % Tensor V_n
    tB{n} = zeros([dim,1]); % Tensor B_n
end

% tic
[L U] = factor(X, rho);
% totalTime = totalTime + toc;
%disp('Time for pre-computation:')
%disp(totalTime)
%% Iterate: Main algorithm
% tic;
for k = 1:max_iter
    % Update tU: quadratic proximal operator
    
    % tic
    q = Xty + rho*(tW(:) - tA(:));
    
    if numSam >= numFea
        u = U \ (L \ q);
    else
        u = lambda*(q - lambda*(X'*(U \ ( L \ (X*q) ))));
    end    
    
    %u = (U \ L) \ q;
    % totalTime = totalTime + toc;
    
    tU = reshape(u, [dim,1]);
    
    % Update tV_n: trace-norm proximal operator
    for n = 1:N
        tV{n} = Fold( prox_nuclear( Unfold(tW-tB{n}, dim, n), alpha/rho/N ), dim, n );
    end
    
    % Update tW: l1-norm proximal operator
    last_tW = tW;
    tSum_uv = tU;
    tSum_ab = tA;
    for n = 1:N
        tSum_uv = tSum_uv + tV{n};
        tSum_ab = tSum_ab + tB{n};
    end
    tW =  prox_l1( (tSum_uv+tSum_ab)/(N+1), beta/(N+1)/rho );
    
    % Update tA
    tA = tA + tU - tW;
    
    % Update tB_n
    for n = 1:N
        tB{n} = tB{n} + tV{n} - tW;
    end
    
    % Check termination
    errList(k) = norm(tW(:)-last_tW(:)) / norm(tW(:));
    
    if errList(k) < epsilon || isnan(errList(k))
        % Revise [Jiaqi Zhang]: print out the number of iterations
        %fprintf('Iteration number %d \n',k) %[Jiaqi Zhang] no output
        % revised end
        break;
    end
end
% time = toc;

%disp('Time for computation:')
%disp(time)

end



