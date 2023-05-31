%% ===================================================
%                cvfun() 
% ===================================================
function [Lambda, TestErr,TestW,TestSigma] = cvfun(cvp, Xten, Xt, Y,epsilon,xi,MaxIter,alpha,early_stop,nstep_wait)
    Lambda = cell(cvp.NumTestSets,1);
%     TrainErr = cell(cvp.NumTestSets,1);
    TestErr = cell(cvp.NumTestSets,1);
    TestW = cell(cvp.NumTestSets,1);
    TestSigma = cell(cvp.NumTestSets,1);
    for i = 1:cvp.NumTestSets
        trIdx = cvp.training(i);
        teIdx = cvp.test(i);
        if ndims(Xten)==3
            Xtrain = Xten(:,:,trIdx);
            Xtest  = Xten(:,:,teIdx);
        elseif ndims(Xten)==4
            Xtrain = Xten(:,:,:,trIdx);
            Xtest  = Xten(:,:,:,teIdx);
        end
        Ytrain = Y(trIdx);
        Ytest  = Y(teIdx);
        Xttrain = Xt(trIdx,:);
        [Lambda{i,1},TestErr{i,1},TestW{i,1},TestSigma{i,1}] = TrainAndPredict(Xtrain,Ytrain, ...
        Xtest,Ytest,Xttrain,epsilon,xi,MaxIter,alpha,early_stop,nstep_wait);    
    end