function [ best ] = cross_validation_lowRank( X, y, setting )

indexTotal = 1:length(y);
numFold    = setting.valFold;
numTest    = length(indexTotal) / numFold;


best.acc = 0;
for alpha = setting.alphaRange
    fprintf('alpha: %d\n', alpha);
    for testFold = 1:numFold
        %% Train--test
        indexTest  = indexTotal((testFold-1)*numTest+1 : testFold*numTest);
        indexTrain = setdiff(indexTotal,indexTest);
        
        tX_test = X(:, :, :, indexTest);
        y_test = y(indexTest, :);
        
        tX_train = X(:, :, :, indexTrain);
        y_train = y(indexTrain, :);
        
        %% Cross validate
        [tW, errlist] = Remurs(tX_train, y_train, alpha, 0, setting.epsilon, 1000);
        
        X_test = reshape(tX_test, [], size(tX_test, 4));
        X_test = X_test';
        
        %% Predict
        y_predict = X_test * tW(:);
        threshold = (max(y)+min(y))/2;;
        y_predict(find(y_predict >= threshold)) = max(y);
        y_predict(find(y_predict <  threshold)) = min(y);
              
        acc(testFold) = length(find(y_predict == y_test)) / length(y_test);
    end
    
    avgAcc = mean(acc);
    
    if avgAcc > best.acc
        best.acc   = avgAcc;
        best.alpha = alpha;
    end
end


