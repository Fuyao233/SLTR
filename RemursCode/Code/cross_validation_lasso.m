function [ best ] = selOptFeaNum( X, y, setting )

indexTotal = 1:length(y);
numFold    = setting.valFold;
numTest    = length(indexTotal) / numFold;

isBetaMax = false;

best.acc = 0;
for beta = setting.betaRange
    fprintf('beta: %d\n', beta);
    if isBetaMax
        fprintf('Reach max beta! Break!!!\n');
        break;
    end
    acc = zeros(1,numFold);
    num = zeros(1,numFold);
    
    for testFold = 1:numFold
        %% Train--test
        indexTest  = indexTotal((testFold-1)*numTest+1 : testFold*numTest);
        indexTrain = setdiff(indexTotal,indexTest);
        
        X_test = X( indexTest, : );
        y_test = y( indexTest, : );
        
        X_train = X( indexTrain, : );
        y_train = y( indexTrain, : );
        %% Cross validation
        [w, funVal] = LeastR(X_train, y_train, beta, []);
        %% Predict
        y_predict = X_test * w;
        threshold = (max(y)+min(y))/2;
        y_predict(find(y_predict >= threshold)) = max(y);
        y_predict(find(y_predict <  threshold)) = min(y);
        
        acc(testFold) = length(find(y_predict == y_test)) / length(y_test);
    end
    avgAcc = mean(acc);
    
    if avgAcc > best.acc
        best.acc   = avgAcc;
        best.beta = beta;
    end
    
end
end

