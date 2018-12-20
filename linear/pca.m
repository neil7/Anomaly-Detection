function pc  = pca(x)
% [pc,sv,n_sv]  = pca(x)
%
% Input:
%   x - Data stored column-vise .
%
% Output:
% pc - Principal components (eigenvectors of the covariance matrix).
%  sv     - Singular values.
%  n_sv - Normalized singular values.

C = cov(x);
[U,D,pc] = svd(C);
sv = diag(D);
n_sv = 100*sv/sum(sv);