%{
Description:
    Cross-validation for EE_Remurs.
    
%}
function [cvTau, cvLambda, cvEpsilon, time] = cv_Prox_Remurs(X, Y, p, tauList, lambdaList, epsilonList, rho, fold, maxIter, minDiff)
    addpath('tensor_toolbox/')
    %addpath('RemursCode/')

    if isempty(maxIter)
        iter = 1000;
    end
    if isempty(minDiff)
        minDiff = 1e-3;
    end
    M = length(p);
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(tauList)
        for j = 1:length(lambdaList)
            for k = 1:length(epsilonList)
                parameterPair{i,j,k} = [tauList(i) lambdaList(j), epsilonList(k)];
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
    
    NUM = length(tauList)*length(lambdaList)*length(epsilonList);
    cv_num = length(tauList)*length(lambdaList)*length(epsilonList);

    dim_flag = 0; % flag to indicate whether the dimension of input is 4 or 5

    % calculate the group of startApprox
    time = [0, 0];
    for f = 1:cvp.NumTestSets
        disp('cvp.NumTestSets:');
        disp(f);
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
        % disp('size of Xtest')
        % disp(size(trainIndex))
        
        if length(size(X)) == 5
            % 5D
            Xtrain = X(:,:,:,:,trainIndex);
            Ytrain = Y(trainIndex,:);
            Xtest = X(:,:,:,:,testIndex);
            Ytest = Y(testIndex,:);
            
            % dim_flag = 5;
        end

        if length(size(X)) == 4
            % 4D
            Xtrain = X(:,:,:,trainIndex);
            Ytrain = Y(trainIndex,:);
            Xtest = X(:,:,:,testIndex);
            Ytest = Y(testIndex,:);
            
            
        end

        if f==1
            [startApprox, start_time] = Prox_Remurs_getStartApprox(double(Xtrain), double(Ytrain));
            s = size(startApprox);
            dim_flag = length(s);
            % disp(s)
            if dim_flag == 3
                startApprox_group = double(zeros(cvp.NumTestSets, s(1), s(2), s(3)));
                startApprox_group(f,:,:,:) = startApprox;
            else
                startApprox_group = double(zeros(cvp.NumTestSets, s(1), s(2), s(3), s(4)));
                startApprox_group(f,:,:,:,:) = startApprox;
            end
            time(1) = time(1) + start_time;
        else
            % [startApprox, start_time] = Prox_Remurs_getStartApprox(double(Xtrain), double(Ytrain));
            s = size(startApprox);
            time(1) = time(1) + start_time;
            if dim_flag == 5
                % startApprox_group = tensor(zeros(cvp.NumTestSets, s(1), s(2), s(3), s(4), s(5)));
                startApprox_group(f,:,:,:,:,:) = startApprox;
            else
                % startApprox_group = tensor(zeros(cvp.NumTestSets, s(1), s(2), s(3), s(4)));
                startApprox_group(f,:,:,:,:) = startApprox;
            end
        end

    end
    time(1) = time(1) / cvp.NumTestSets;

    for t = 1:cv_num
        testErr = 0.0;
        for f = 1:cvp.NumTestSets
            % disp('cvp.NumTestSets:');
            % disp(cvp.NumTestSets);
            pars = parameterPair{t};
            if fold > 1
                trains = cvp.training(f);
                tests =  cvp.test(f);
            else
                trains = cvp.trains;
                tests = cvp.tests;
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
            % disp('size of Xtest')
            % disp(size(trainIndex))

            if length(size(X)) == 5
                % 5D
                Xtrain = X(:,:,:,:,trainIndex);
                Ytrain = Y(trainIndex,:);
                Xtest = X(:,:,:,:,testIndex);
                Ytest = Y(testIndex,:);
            end

            if length(size(X)) == 4
                % 4D
                Xtrain = X(:,:,:,trainIndex);
                Ytrain = Y(trainIndex,:);
                Xtest = X(:,:,:,testIndex);
                Ytest = Y(testIndex,:);
            end

            % disp('size of Xtrain')
            % disp(size(Xtrain))
            % disp('prox_remurs233');

            if dim_flag == 5
                [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), pars(3), rho, maxIter, minDiff, startApprox_group(f,:,:,:,:,:));
            else
              [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), pars(3), rho, maxIter, minDiff, startApprox_group(f,:,:,:,:));
%               [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), pars(3), rho, maxIter, minDiff);
            end
            time(2) = time(2) + iteration_time(2);
            if iteration_time(2) > 1
                disp('???');
            end

            % compute MSE
            
            if length(size(X)) == 5
                Xtest = permute(Xtest,[5 1 2 3 4]); % 5D    
            end

            if length(size(X)) == 4
                Xtest = permute(Xtest,[4 1 2 3]); % 4D
            end

            % disp(size(tensor(Xtest)))
            % disp(size(tensor(estimatedW)))
            
            predY = ttt(tensor(Xtest), tensor(estimatedW), 2:M+1, 1:M); 
            if fold > 1
                testErr = testErr + norm(tensor(predY.data, [cvp.TestSize(f) 1]) - tensor(Ytest)) / cvp.TestSize(f);
            else
                testErr = testErr + norm(tensor(predY.data, [cvp.num_ones 1]) - tensor(Ytest)) / length(cvp.num_ones);
            end
        end
        testErr = testErr / fold;

        % disp(NUM);
        % disp(pars);
        NUM = NUM - 1;
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
        fprintf('cv_Prox:[%d/%d]\t testErr:%f \t minErr:%f \n', t, cv_num, testErr, minErr)
    end
    time(2) = time(2) / cv_num;

    cvTau = bestPair(1);
    cvLambda = bestPair(2);
    cvEpsilon = bestPair(3);
    fprintf('cvTau : %f; cvLambda : %f; cvEpsilon : %f\n', cvTau, cvLambda, cvEpsilon)
   
        