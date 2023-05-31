function x = prox_nuclear(v, lambda)
%   Evaluates the proximal operator of the nuclear norm at v
%   (i.e., the singular value thresholding operator).

    [U,S,V] = svds(v);
    %[U,S,V] = MySVD(v);
    x = U*diag(prox_l1(diag(S), lambda))*V';
end
