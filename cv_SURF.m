%{
Description:
    Cross-validation for SURF.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [cvAlpha, cvEpsilon, cvR, cv_time] = cv_SURF(X, Xvec, Y, p, alphaList, epsilonList,...
    RList, fold, absW)
    addpath('SURF_code/')
    addpath('tensor_toolbox/')

    cv_time = 0;

    if isempty(absW)
        absW = 0.1;
    end
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(alphaList)
        for j = 1:length(epsilonList)
            for k = 1:length(RList)
                parameterPair{i,j,k} = [alphaList(i) epsilonList(j) RList(k) epsilonList(j)^2/2];
            end
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
    cv_num = length(alphaList)*length(epsilonList)*length(RList);
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
            Xtrain = X(:,:,:,trainIndex);
            Xvectrain = Xvec(trainIndex,:);
            Ytrain = Y(trainIndex);
            Xtest = X(:,:,:,testIndex);
            Ytest = Y(testIndex);
            % training
            R = pars(3);
            estimatedW = zeros(R,prod(p));
            res = Ytrain;

            for r =1:R
                [W_r, residual] = MyTrain(Xtrain, Xvectrain, double(res), pars(2), pars(4), pars(1), absW);
                res = residual;
                estimatedW(r,:) = W_r;
            end
            
            % response error
            estimatedWVec = zeros(1,prod(p)); 
            for r = 1:R
                estimatedWVec = estimatedWVec + estimatedW(r,:);
            end
            if fold > 1
                predY = zeros(cvp.TestSize(f), 1);
            else
                predY = zeros(length(cvp.num_ones), 1);
            end
            vecX = tenmat(Xtest, 4); % NOTES: for 3D variates
            vecX = vecX.data;

            if fold > 1
                for i = 1:cvp.TestSize(f)
                    predY(i) = vecX(i,:) * estimatedWVec';
                end
                testErr = norm(tensor(predY - Ytest)) / cvp.TestSize(f);
            else
                for i = 1:length(cvp.num_ones)
                    predY(i) = vecX(i,:) * estimatedWVec';
                end
                testErr = norm(tensor(predY - Ytest)) / length(cvp.num_ones);
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
        fprintf('cv_SURF:[%d/%d]\t testErr:%f \t minErr:%f \n', t, cv_num, testErr, minErr)
    end
    cvAlpha = bestPair(1);
    cvEpsilon = bestPair(2);
    cvR = bestPair(3);
    fprintf('cvAlpha : %f; cvEpsilon : %f; cvR : %d\n', cvAlpha, cvEpsilon, cvR)
   
        