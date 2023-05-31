%{
Description:
    Cross-validation for EE_Remurs to study if Prox_Remurs is sensitive to the epsilon
    
%}
function [bestPair, time] = cv_Prox_Remurs_epsilon(X, Y, p, tauList, lambdaList, epsilonList, rho, fold, maxIter, minDiff,epsilon_eig)
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
            parameterPair{i,j} = [tauList(i) lambdaList(j)];
        end
    end
    % cross-validation initialization  

    N = size(X);
    N = N(end);
    cvp = cvpartition(N, 'Kfold', fold);
    NUM = length(tauList)*length(lambdaList);
    cv_num = length(tauList)*length(lambdaList);

    dim_flag = 0; % flag to indicate whether the dimension of input is 4 or 5

    % calculate the group of startApprox
    time = zeros(length(epsilonList), 2);
    for f = 1:cvp.NumTestSets
        disp('cvp.NumTestSets:');
        disp(f);
        trains = cvp.training(f);
        tests =  cvp.test(f);
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
        
        
%         if dim_flag == 3
%             startApprox_group = double(zeros(length(epsilonList), cvp.NumTestSets, s(1), s(2), s(3)));
%         else
%             startApprox_group = double(zeros(length(epsilonList), cvp.NumTestSets, s(1), s(2), s(3), s(4)));
%         end

        for epsilon_index = 1:length(epsilonList)
            disp('epsilon:')
            disp(epsilonList(epsilon_index))
            if epsilonList(epsilon_index) > 0 
                [startApprox, start_time] = Prox_Remurs_getStartApprox(double(Xtrain), double(Ytrain), epsilonList(epsilon_index));
            else
                [startApprox, start_time] = Prox_Remurs_getStartApprox(double(Xtrain), double(Ytrain), epsilon_eig{f}(1));
            end
            s = size(startApprox);
            time(epsilon_index, 1) = time(1) + start_time;
            if dim_flag == 3
                % startApprox_group = tensor(zeros(cvp.NumTestSets, s(1), s(2), s(3), s(4), s(5)));
                startApprox_group(epsilon_index, f,:,:,:) = startApprox;
            else
                % startApprox_group = tensor(zeros(cvp.NumTestSets, s(1), s(2), s(3), s(4)));
                startApprox_group(epsilon_index, f,:,:,:,:) = startApprox;
            end
        end

    end
    time(:,1) = time(:, 1) / cvp.NumTestSets;

    for t = 1:cv_num
        testErr = 0.0;
        
        for epsilon_index = 1:length(epsilonList)
            % TODO: 对每一组tau和lambda计算五折平均损失并记录各自epsilon的最佳损失及cv时间
            for f = 1:cvp.NumTestSets
                % disp('cvp.NumTestSets:');
                % disp(cvp.NumTestSets);
                pars = parameterPair{t};
                trains = cvp.training(f);
                tests =  cvp.test(f);
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
                    [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), epsilonList(epsilon_index), rho, maxIter, minDiff, startApprox_group(epsilon_index,f,:,:,:,:,:));
                else
                    [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), epsilonList(epsilon_index), rho, maxIter, minDiff, startApprox_group(epsilon_index,f,:,:,:,:));
    %               [estimatedW, ~, iteration_time] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), pars(3), rho, maxIter, minDiff);
                end
                time(epsilon_index,2) = time(epsilon_index,2) + iteration_time;

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
                testErr = testErr + norm(tensor(predY.data, [cvp.TestSize(f) 1]) - tensor(Ytest)) / cvp.TestSize(f);
            end
            testErr = testErr / fold;

            % disp(NUM);
            % disp(pars);
            NUM = NUM - 1;
            % update the best setting of parameters
            
            if t == 1
                minErr = testErr;
                bestPair{epsilon_index} = parameterPair{t};
            else
                if testErr < minErr
                    minErr(epsilon_index) = testErr;
                    bestPair{epsilon_index} = parameterPair{t};
                end
            end
            
%             fprintf('cv_Prox:[%d/%d]\t testErr:%f \t minErr:%f \n, epsilon:%f \n', t, cv_num, testErr, minErr, epsilon)
            fprintf('cv_Prox:[%d/%d]\t epsiolon:%f', t, cv_num, epsilonList(epsilon_index))
        end
    end
    time(:,2) = time(:,2) / cv_num;

    % cvTau = bestPair(1);
    % cvLambda = bestPair(2);
    % cvEpsilon = bestPair(3);
    % fprintf('cvTau : %f; cvLambda : %f; cvEpsilon : %f\n', cvTau, cvLambda, cvEpsilon)
    % Res.bestPair = bestPair;
    % Res.time = time;


   
        