function [W,eigenvalues,sigmasquare,mu] = RegularPCA(data,q)
%RegularPCA.m
%   Implement probabilistic PCA algorithm from C.M. Bishop 1999 
%    Microsoft Research

%INPUT: Data - d-by-N matrix of N d-dimensional data points
%        OPTIONAL
%       q - the maximum dimensionality of the latent space (the maximum
%         number of retained eigenvalues/eigenvectors), defaults to q=d-1
%
%OUTPUT: W - maximum a posteriori solution for W, where W is a low-rank
%         representation of the data covariance matrix S ( S = cov(data) )
%           ... columns are principal eigenvectors
%        eigenvalues - vector of eigenvalues (equivalent to the variance
%           along the axis created by the corresponding eigenvector)
%        C - the approximate covariance matrix extrapolated from the
%         matrix W
%           C = W*W'+sigmasquare*eye(d)
%        sigmasquare - the average variance of the discarded dimensions, for
%         reconstruction of the covariance matrix C
%  
% this implementation of PCA assumes a latent-variable model of the form
%    data(:,ii) = W*x(:,ii)+mu+E  ... ii=1:N
%      where mu = mean(data,2); ... the mean of each data dimension
%      E is the error (zero-mean gaussian)
%      x is the latent space representation with prior N(0,I-q)
%        I-q is the identity matrix with q = d-1 dimensions
%      W is the linear transformation matrix that transforms the data
%       from the latent space to the original space
%      
%        x(:,ii) = pinv(W)*(data(:,ii)-mu)
% 
%Created: 2017/05/24
% Byron Price
%Updated: 2017/07/25
%By: Byron Price

% example data
% d = 10;N = 500;data = zeros(d,N);
% for ii=1:N
%     for jj=1:4
%         data(jj,ii) = normrnd(0,1);
%     end
% end
% for ii=1:N
%     for jj=5:d
%         data(jj,ii) = normrnd(0,0.4);
%     end
% end
% dataMu = mean(data,2);
% data = data-repmat(dataMu,[1,N]);
warning('off','all');

[d,N] = size(data);

if nargin == 1
    q = d-1;
end

mu = mean(data,2);
data = data-repmat(mu,[1,N]);

S = cov(data');
[V,D] = eig(S);
start = d-q+1;
eigenvals = diag(D);
meanEig = mean(eigenvals(1:start-1));
W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
W = fliplr(W);
sigmasquare = meanEig;
eigenvalues = flipud(eigenvals(start:end));
end