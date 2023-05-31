function [ best ] = selOptFeaNum( X, y, setting )

indexTotal = 1:length(y);
numFold    = setting.valFold;
numTest    = length(indexTotal) / numFold;


best.acc = 0;
for logAlpha = setting.alphaRange
    alpha = 2^logAlpha;
    isBetaMax = false;
    for logBeta = setting.betaRange
        beta = 2^logBeta;
        fprintf('alpha: %d, beta: %d\n', logAlpha, logBeta);
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
            
            opt.rsL2 = beta * 2;
            
            [w, funVal] = LeastR(X_train, y_train, alpha, opt);
            
            y_predict = X_test * w;
            threshold = (max(y)+min(y))/2;
            y_predict(find(y_predict >= threshold)) = max(y);
            y_predict(find(y_predict <  threshold)) = min(y);
            
            num(testFold) = length(find(w(:) ~= 0));
            if num(testFold) < 50
                isBetaMax = true;
                break;
            end
            
            acc(testFold) = length(find(y_predict == y_test)) / length(y_test);
        end
        avgAcc = mean(acc);
        
        if ~isBetaMax & avgAcc > best.acc
            best.acc   = avgAcc;
            best.alpha = alpha;
            best.beta  = beta * 2;
        end
    end
end
end

