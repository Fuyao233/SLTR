%{
Description:
    Sparse Higher-Order Tensor Regression Models with Automatic Rank Explored 

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [estimatedW, errSeq, time, startApprox, mid] = Prox_Remurs(X, Y, tau, lambda, epsilon, rho, maxIter, minDiff, startApprox)
    % 2023.2.28 Add parameter 'startApprox', check the number of input parameters when call the function
    % if 'nargin == 8', then it means that the startApprox is included 

    totalTime = 0;
    %% Initializations
    %rho = 1; % TODO: set to 1 at present
    p = size(X);
    prodP = prod(p(1:end-1));
    N = p(end);
    dims = ndims(X);
    M = dims - 1;
    if nargin == 8
        disp('No initial solution!');
        
        
        % before code optimization
%         vecX = Unfold(X, p, dims);
%         disp('Pre-computation')
%         tic;
%         mid = vecX' * vecX;
%         startApprox = ( (mid + epsilon * eye(prodP)) \ eye(prodP) ) * vecX' * Y;
%         startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
%         startApprox(isnan(startApprox)) = 0;
%         totalTime = toc;
%         disp('Time for pre-computation:')
%         disp(totalTime)
%         disp('size of startApprox:');
%         disp(size(startApprox))
            
        % after code optimization
        vecX = Unfold(X, p, dims);
        numFea   = size(vecX,2);
        numSam   = size(vecX,1);
        disp('Pre-computation')
        tic;
        [L U] = factor(vecX, 1);
        q = vecX' * Y;
        if numSam >= numFea
            startApprox = U \ (L \ q);
        else
            startApprox = q - vecX' * ( U \ (L \ vecX * q));
        end
        
        startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
        startApprox(isnan(startApprox)) = 0;
        totalTime = toc;
        disp('Time for pre-computation:')
        disp(totalTime)

    else
        disp('Initial solution!');
    end
    
    % operator_time = zeros(1, 7);
    disp(size(startApprox))
    tic
    %% Main Procedure
    for m=1:M
%         disp('m')
%         disp(m)
%         disp('M')
%         disp(M)

        z_m = Unfold(startApprox, p(1:M), m);
        for i = 1:4
            W_t{i} = z_m;
        end
        W_m = z_m;
        lastW = W_m;

        % operator_time(1) = operator_time(1) + toc;
        for t=1:maxIter

            a_t{1} = l1_prox(W_t{1}, 4*lambda);
            % operator_time(2) = operator_time(2) + toc;
            nz1 = length(find(a_t{1} == 0));
            na1 = numel(a_t{1});
            % fprintf('The rate of zeros in a_t{1}: %.4f\n', nz1/na1);
            

            a_t{2} = inf_set_prox(W_t{2}, z_m, 4*lambda);
            % operator_time(3) = operator_time(3) + toc;
            nz2 = length(find(a_t{2} == 0));
            na2 = numel(a_t{2});
            % fprintf('The rate of zeros in a_t{2}: %.4f\n', nz2/na2);


            a_t{3} = nuclear_prox(W_t{3}, 4*tau);
            % operator_time(4) = operator_time(4) + toc;
            nz3 = length(find(a_t{3} == 0));
            na3 = numel(a_t{3});
            % fprintf('The rate of zeros in a_t{3}: %.4f\n', nz3/na3);
                        
            
            a_t{4} = spec_set_prox(W_t{4}, z_m, 4*tau);
            % operator_time(5) = operator_time(5) + toc;
            nz4 = length(find(a_t{4} == 0));
            na4 = numel(a_t{4});
            % fprintf('The rate of zeros in a_t{4}: %.4f\n', nz4/na4);
            
            
            a = (a_t{1} + a_t{2} + a_t{3} + a_t{4}) / 4; % a^t
            for i = 1:4
                W_t{i} = W_t{i} + rho*(2*a-W_m-a_t{i});
            end
            W_m = W_m + rho*(a-W_m);
            
           %% Terminatio  check
            errSeq(t) = norm(tensor(lastW-W_m)) / prod(p(1:M));
            if  errSeq(t) <= minDiff
                %fprintf('Algorithm terminates at the %d-th iteration\n', t)
                fprintf('Last_num_iter:%d\n', t);

                AT.a1 = a_t{1};
                AT.a2 = a_t{2};
                AT.a3 = a_t{3};
                AT.a4 = a_t{4};

%                 save(sprintf('UCF-101/res/operators/AT0.mat'), 'AT');
                break
            end
            lastW = W_m;
            

            % operator_time(6) = operator_time(6) + toc;
        end
        fprintf('num_iter:%d\n', t);
        
        if m == 1
            estimatedW = Fold(W_m, p(1:M), m);
        else
            estimatedW = estimatedW +  Fold(W_m, p(1:M), m);
        end
    end
    estimatedW = estimatedW / M;
    iter_time = toc;
    time = [totalTime, iter_time];


    % operator_time(7) = operator_time(7) + toc;
    disp('Time for iteration:')
    disp(time)
    %predY = zeros(N, 1);
    %for i = 1:N
    %    predY(i) = vecX(i,:) * estimatedW;
    %end
    %mse = norm(tensor(predY-Y))/N;
    %estimatedW = reshape(estimatedW, p(1:M));

    
    
    