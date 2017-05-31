function [W,eigenvalues,sigmasquare,C,qeff] = BayesianPCA(data,q)
%BayesianPCA.m
%   Implement Bayesian PCA algorithm from C.M. Bishop 1999 
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
%        qeff - effective dimensionality of the latent space
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
%        x(:,ii) = inv(W)*(data(:,ii)-mu)
% 
%Created: 2017/05/24
% Byron Price
%Updated: 2017/05/31
%By: Byron Price
warning('off','all');

[d,N] = size(data);

if nargin == 1
    q = d-1;
end

numIter = 5000;

alpha = zeros(q,1);

mu = mean(data,2);
data = data-repmat(mu,[1,N]);

% [V,D] = eig(S);
% W = V(:,2:end)*sqrtm(D(2:end,2:end)-D(1,1)*eye(q));
% eigvals = diag(D);maxeigval = max(eigvals);
estSigma = var(data(:))/2;

if d*N <= 1e5
    S = cov(data');
    [V,D] = eig(S);
    start = d-q+1;
    eigenvals = diag(D);
    meanEig = mean(eigenvals(1:start-1));
    W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
    W = fliplr(W);
else
    W = normrnd(0,1,[d,q]);
end

for jj=1:q
    alpha(jj) = d/(norm(W(:,jj),'fro')^2);
end

expectedMean = zeros(q,N);
expectedCov = zeros(q,q,N);
% BAYESIAN PCA (BISHOP) EM ... with constraints
% figure();
for ii=2:numIter
   prevW = W;prevSigma = estSigma;
   M = W'*W+estSigma.*eye(q);
   Minv = inv(tril(M));
   
   expectedMean = Minv*W'*data;
   for jj=1:N
      expectedCov(:,:,jj) = estSigma.*Minv+(expectedMean(:,jj)*expectedMean(:,jj)'); 
   end
   
   A = diag(alpha);
   
   W = (data*expectedMean')*inv(triu(sum(expectedCov,3)+estSigma.*A));
   
   for jj=1:q
      alpha(jj) = d/(W(:,jj)'*W(:,jj)); 
   end
   
   temp = 0;
   for jj=1:N
       temp = temp+norm(data(:,jj),'fro')^2-2*(expectedMean(:,jj)')*(W')*data(:,jj)+...
           trace(squeeze(expectedCov(:,:,jj))*(W')*W);
   end
   estSigma = temp./(N*d);

    if abs(estSigma-prevSigma) < 1e-6 && sum(abs(W(:)-prevW(:))) < 1e-6
        break;
    end
%     scatter(ii,sum(sum(abs(S-(W*W'+estSigma.*eye(d))))));hold on;pause(0.01);
end

fprintf('Number of iterations: %d\n',ii);
sigmasquare = estSigma;
result = [];qeff = 0;temp = [];
for jj=1:q
    columnNorm = norm(W(:,jj),'fro');
    if columnNorm < 1e-3
    else
        qeff = qeff+1;
        result = [result,W(:,jj)./columnNorm];
        temp = [temp,W(:,jj)];
    end
end
W = temp;
C = W*W'+sigmasquare*eye(d);

eigenvalues = zeros(qeff,1);
if d*N <= 1e5
    for ii=1:qeff
        eigenvalues(ii) = result(:,ii)'*S*result(:,ii); 
    end
else
   eigenvalues = diag(W'*W+sigmasquare.*eye(qeff));
end 

fprintf('Effective dimensionality: %d\n',qeff);
end

%  for constrained EM PCA ... no orthogonal ambiguity
% for iter=1:maxIter
%  H = tril(W'*W)\(W'*data);
%  W = (data*H')/triu(H*H');
% end

