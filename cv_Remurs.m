%{
Description:
    Cross-validation for Remurs.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [cvAlpha, cvBeta, cv_time] = cv_Remurs(X, Y, p, alphaList, betaList, fold, iter, epsilon)
    addpath('tensor_toolbox/')
    addpath('RemursCode/')
    cv_time = 0;

    if isempty(iter)
        iter = 1000;
    end
    if isempty(epsilon)
        diff = 1e-3;
    end
    M = length(p);
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(alphaList)
        for j = 1:length(betaList)
                parameterPair{i,j} = [alphaList(i) betaList(j)];
        end
    end
    % cross-validation initialization  
    N = size(X);
    N = N(end);
    if fold > 1
      cvp = cvpartition(N, 'Kfold', fold);
    else
      cvp.NumTestSets = 1;
    end
    cv_num = length(alphaList)*length(betaList);
    for t = 1:cv_num
        testErr = 0.0;
        for f = 1:cvp.NumTestSets
            pars = parameterPair{t};
            if fold > 1
                trains = cvp.training(f);
                tests =  cvp.test(f);
            else
                % randomly set training index : test index = 8 : 2
                trains = zeros(N,1);
                tests = zeros(N, 1);
                num_ones = round(0.2 * N);
                ones_idx = randperm(N, num_ones);
                tests(ones_idx) = 1;
                trains(ones_idx) = -1;
                trains = trains + 1;

                cvp.tests = tests;
                cvp.trains = trains;
                cvp.num_ones = num_ones;
            end
            trainIndex = [];
            testIndex = [];
            for i = 1:N
                if trains(i) == 1
                    trainIndex = [trainIndex i];
                end
                if tests(i) == 1
                    testIndex = [testIndex i];
                end
            end
            % 3-D variates 
            if length(size(X)) == 5
                % 5D
                % disp('5D')
                Xtrain = X(:,:,:,:,trainIndex);
                Ytrain = Y(trainIndex,:);
                Xtest = X(:,:,:,:,testIndex);
                Ytest = Y(testIndex,:);
                % disp(size(Xtrain))
            end

            if length(size(X)) == 4
                % 4D
                % disp('4D')
                Xtrain = X(:,:,:,trainIndex);
                Ytrain = Y(trainIndex,:);
                Xtest = X(:,:,:,testIndex);
                Ytest = Y(testIndex,:);
            end
            

            % disp(size(Xtrain))
            tic
            [estimatedW, ~] = Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), epsilon, iter);
            cv_time = cv_time + toc;
            % compute MSE
            % disp('before')
            % disp(size(Xtest))
            % if length(size(X)) == 5
            %     disp('5D')
            %     Xtest = permute(Xtest,[5 1 2 3 4]); % 5D    
            % end

            % if length(size(X)) == 4
            %     Xtest = permute(Xtest,[4 1 2 3]); % 4D
            % end

            % disp(size(Xtest))
            % disp(size(estimatedW))

            predY = ttt(tensor(Xtest), tensor(estimatedW), 1:M, 1:M);
            if fold > 1
                testErr = testErr + norm(tensor(predY.data, [cvp.TestSize(f) 1]) - tensor(Ytest)) / cvp.TestSize(f);
            else
                testErr = testErr + norm(tensor(predY.data, [cvp.num_ones 1]) - tensor(Ytest)) / length(cvp.num_ones);
            end
        end
        testErr = testErr / fold;
        % update the best setting of parameters
        if t == 1
            minErr = testErr;
            bestPair = parameterPair{t};
        else
            if testErr < minErr
                minErr = testErr;
                bestPair = parameterPair{t};
            end
        end
        fprintf('cv_Remurs:[%d/%d]\t testErr:%f \t minErr:%f \n', t, cv_num, testErr, minErr)
    end
    cv_time = cv_time / cv_num;
    cvAlpha = bestPair(1);
    cvBeta = bestPair(2);
    fprintf('cvAlpha : %f; cvBeta : %f\n', cvAlpha, cvBeta)
   
        