pList  = [...
    [10 10 5];...
    [15 15 5];...
    [20 20 5];...
    [25 25 5];...
    [30 30 5];...
    [35 35 5];...
    [40 40 5];...
    ];
N = [40 90 160 250 360 490 640]; %0.08

%pList  = [[10 10 5];];
%N = [50];


warning off;

for i = 1:length(pList)
    
    %% Generate simulated datasets
    %clear X W Y Xvec Wvec invertX estimatedW fit;

    addpath('tensor_toolbox/');
    % parameters for generating datasets。
    % options.p = [20 20 20];
    options.p = pList(i,:);
    %options.N = 100;
    options.N = N(i);
    options.R = 5;
    options.sparsity = 0.2; 
    options.noise_coeff = 0.1;
    M = length(options.p);
    %{
        X -- a tensor with shape N x p1 x p2 x...x pM 
        W -- a tensor with shape p1 x p2 x...x pM 
        Y -- a trensor with shape N
        Xvec -- a matrix with shape N x (p1 * p2 *...* pM)
        Wvec -- a vector with shape (p1 * p2 *..* pM) x 1 
        invertX -- a tensor with shape p1 x p2 x...x pM x N
    %}
    %[X, W, Y, Xvec, Wvec, invertX] = generateData(options);
    [X, W, Y, Xvec, Wvec, invertX] = sparseGenerate(options);
    disp(options)

    %% Experiment settings
    repeat =2;

    X = double(invertX);
    p = size(X);
    dims = ndims(X);
    vecX = Unfold(X, p, dims);
    tic;
    epsilon = abs(max(eig(vecX' * vecX)));
    epsilon_time = toc;
    Epsilon_data{i} = [epsilon epsilon_time];
end