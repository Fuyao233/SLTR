%{
function [optiLamb,SolPath,TestErrInt,TestErrRSD, stop] = MyInterp(Lambda,TestErr,zero_stop)
    SolPath = cell2mat(Lambda');
    SolPath = unique(SolPath,'stable');
    for i = 1:length(Lambda)
        x = Lambda{i,1};
        y = TestErr{i,1};
        [xu,idx] = unique(x,'last');
        yu = y(idx); 
       %% Square Interpolation
       if length(xu)>=2
           errSqrt = interp1(xu,yu,SolPath,'linear','extrap');
           TestErrInt(i,:) = errSqrt;
           if sum(TestErrInt(i,:)<0)>=1
               fprintf('Interpolation error requires nonnegative value in each point.\n');
               break;
           end
       else
           TestErrInt(i,:) = ones(1,length(SolPath)) * yu;
       end
       %% Linear Interpolation
%        TestErrInt(i,:) = interp1(xu,yu,SolPath,'linear','extrap');
%        TrainErrInt(i,:) = interp1(xu,zu,SolPath,'linear','extrap');
    end
    errm = mean(TestErrInt,1);
    [errmin,Ind] = min(errm);
    TestErrRSD = sqrt(errmin);
    optiLamb = SolPath(Ind);
    if zero_stop==1 && Ind==1
        stop = 1;
    else
        stop = 0;
    end
    fprintf('The average test error of K-fold is %g \n',TestErrRSD);
    end
    %}

function [optiLamb,SolPath,TestErrInt,TestErrRSD,absW] = MyInterp(Lambda,TestErr,tew,sigma)
    SolPath = cell2mat(Lambda');
    SolPath = unique(SolPath,'stable');
    for i = 1:length(Lambda)
        x = Lambda{i,1};
        y = TestErr{i,1};
        [xu,idx] = unique(x,'last');
        yu = y(idx); 
       %% Square Interpolation
       if length(xu)>=2
           errSqrt = interp1(xu,sqrt(yu),SolPath,'linear','extrap');
           TestErrInt(i,:) = errSqrt.^2;
           if sum(TestErrInt(i,:)<0)>=1
               fprintf('Interpolation error requires nonnegative value in each point.\n');
               break;
           end
       else
           TestErrInt(i,:) = ones(1,length(SolPath)) * yu;
       end
       %% Linear Interpolation
%        TestErrInt(i,:) = interp1(xu,yu,SolPath,'linear','extrap');
%        TrainErrInt(i,:) = interp1(xu,zu,SolPath,'linear','extrap');
    end
    errm = mean(TestErrInt,1);
    [errmin,Ind] = min(errm);
    TestErrRSD = sqrt(errmin);
    optiLamb = SolPath(Ind);
    fprintf('The average test error of K-fold is %g \n',TestErrRSD);
%% Compute the average L1 norm of W
    absW = 0;
    p = 0;
    for i=1:5
        idxx = find(optiLamb==Lambda{i,1});
        if ~isempty(idxx)
            wt = cellfun(@transpose,tew{i,1}(max(idxx),:),'un',0);
            wt = cpdgen(wt) * sigma{i,1}(max(idxx));
            absW = absW + sum(abs(wt(:)));
            p = p+1;
        end
    end
    absW = absW/p;
