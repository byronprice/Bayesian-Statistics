function [W,sigmasquare,C,qeff] = BayesianPCA(data)
%BayesianPCA.m
%   Implement Bayesian PCA algorithm from C.M. Bishop 1999 
%    Microsoft Research

%INPUT: Data - d-by-N matrix of N d-dimensional data points
%
%OUTPUT: W - maximum a posteriori solution for W, where W is a lower-rank
%         representation of the data covariance matrix S ( S = cov(data) )
%        C - the approximate covariance matrix extrapolated from the
%         matrix W
%           C = W*W'+sigmasquare*eye(d)
%        sigmasquare - the variance of the discarded dimensions, for
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
%Created: 2017/05/24
% Byron Price
%Updated: 2017/05/24
%By: Byron Price

[d,N] = size(data);
q = d-1;

S = cov(data');

numIter = 500;

alpha = zeros(q,1);
estSigma = zeros(numIter,1);

mu = mean(data,2);
[V,D] = eig(S);
W = V(:,2:end)*sqrtm(D(2:end,2:end)-D(1,1)*eye(q));
estSigma(1) = 1;
figure();plotwb(W);

% BAYESIAN PCA (BISHOP) EM
for ii=2:numIter
   prevW = W;
   M = W'*W+estSigma(ii-1)*eye(q);
   
   xn = zeros(q,N);
   xnxnt = zeros(q,q,N);
   for jj=1:N
      xn(:,jj) = (M\W')*(data(:,jj)-mu);
      xnxnt(:,:,jj) = estSigma(ii-1)*M+xn(:,jj)*xn(:,jj)';
   end
   
   for jj=1:q
      alpha(jj) = d/(norm(W(:,jj))^2); 
   end
   A = diag(alpha);
    
   temp1 = 0;temp2 = 0;
   for jj=1:N
       temp1 = temp1+(data(:,jj)-mu)*xn(:,jj)';
       temp2 = temp2+squeeze(xnxnt(:,:,jj));
   end
   temp2 = temp2+estSigma(ii-1)*A;
   
   W = temp1*inv(temp2);
   
   temp = 0;
   for jj=1:N
       temp = temp+norm(data(:,jj)-mu)^2-2*xn(:,jj)'*W'*(data(:,jj)-mu)+...
           trace(squeeze(xnxnt(:,:,jj))*W'*W);
   end
   estSigma(ii) = temp./(N*d);

    if abs(estSigma(ii)-estSigma(ii-1)) < 1e-5 && abs(sum(abs(W(:)))-sum(abs(prevW(:)))) < 1e-5
        break;
    end
end

sigmasquare = estSigma(ii);
figure();plotwb(W);
tempW = W;result = [];
for jj=1:q
    if sum(abs(W(:,jj))) < 1e-6
    else
        result = [result,jj];
    end
end
W = tempW(:,result);
qeff = length(result);

C = W*W'+sigmasquare*eye(d);
totalError = sum(sum(abs(C-S)))./numel(S);
fprintf('Error in reconstruction of covariance matrix S from W: %3.2f\n',totalError);
fprintf('Effective dimensionality: %d\n',qeff);
end

