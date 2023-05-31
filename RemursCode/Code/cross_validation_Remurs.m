function [ best ] = cross_validation_Remurs(X, y, M, setting, epsilon)

indexTotal = 1:length(y);
numFold    = setting.valFold;
numTest    = length(indexTotal) / numFold;

isBetaMax = false;

best.acc = 0;
for alpha = setting.alphaRange
    isBetaMax = false;
    for beta = setting.betaRange
        if isBetaMax
            fprintf('Reach max beta! Break!!!\n');
            break;
        end
        
        isTooSmallBeta = false;
        
        fprintf('alpha: %d, beta: %d\n', alpha, beta);
        
        acc = 100*zeros(1,numFold);
        num = zeros(1,numFold);
        
        for testFold = 1:numFold
            %% Train--test
            indexTest  = indexTotal((testFold-1)*numTest+1 : testFold*numTest);
            indexTrain = setdiff(indexTotal,indexTest);
            
            tX_test = X(:, :, :, indexTest);
            y_test = y(indexTest, :);
            
            tX_train = X(:, :, :, indexTrain);
            y_train = y(indexTrain, :);
            %% Cross validation
            [tW, errlist] = Remurs(tX_train, y_train, alpha, beta, setting.epsilon, 1000);
           
            X_test = reshape(tX_test, [], size(tX_test, 4));
            X_test = X_test';
            
            %% Predict
            %{
            y_predict = X_test * tW(:);
            threshold = (max(y)+min(y))/2;
            y_predict(find(y_predict >= threshold)) = max(y);
            y_predict(find(y_predict <  threshold)) = min(y);
            %}
            y_predict = ttt(tensor(X_test), tensor(tW(:)), 1:M, 1:M);
            
            num(testFold) = length(find(tW(:) ~= 0));
            %% Early stop if too few voxels are selected.
            if num(testFold) < setting.nVoxels * setting.minFeaPercent
                isBetaMax = true;
                break;
            end
            
            %% Break if too many voxels are selected
            if num(testFold) > setting.nVoxels * setting.maxFeaPercent
                fprintf('Too small beta!!!\n');
                isTooSmallBeta = true;
                break;
            end
            
            acc(testFold) = norm(tensor(y_predict.data, [length(y_test), 1]) - tensor(y_test)) / length(y_test);
                
        end

        avgAcc = mean(acc);
        
        if ~isBetaMax & ~isTooSmallBeta & avgAcc < best.acc
            best.acc   = avgAcc;
            best.alpha = alpha;
            best.beta  = beta;
        end
    end
end
end

