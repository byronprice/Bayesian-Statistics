function [piParam,mu,sigma,alpha] = GaussMixtureEM(data,K)
%GaussMixtureEM.m
%   Fit latent variable Normal-Multinomial model for Gaussian mixture data 
%    using the EM algorithm. Data is assumed to be a mixture of K Gaussian
%    distributions
%     e.g. some subset of data ~ Normal(mu(1),sigma(1)) with probability piParam(1)
%          some other subset of data ~ Normal(mu(2),sigma(2)) with probability
%          piParam(2)
%INPUT: data - observed data, input as a matrix, N-by-d, where N is the
%        number of observations and d is the dimensionality of each observation
%OUTPUTS:
%       piParam - probability that observed data point comes component 1,
%         2,..., K-1
%       mu - cell array with mean of each component
%       sigma - cell array with covariance of each component
%       alpha - log probability that each datapoint comes from a given
%        component
%
%Created: 2018/10/22
% By: Byron Price
%Updated: 2020/01/02

% test code with OldFaithfulGeyser dataset
%  GaussMixtureEM(data,2); 
% make sure the data is 272-by-2 columns, with eruption time in minutes
% and wait time to next eruption in minutes

% check size of data
[N,d] = size(data);

% get log of data
logData = log(data); % uses complex numbers if data less than zero

% initialize parameters
piParam = (1./K).*ones(K,1);

% split data in quadrants
Q = quantile(data(:,1),linspace(0,1,K+1));

mu = cell(K,1);
sigma = cell(K,1);

for ii=1:K
   currentData = data(data(:,1)>=Q(1,ii) & data(:,1)<Q(1,ii+1),:);
   mu{ii} = mean(currentData)';
   sigma{ii} = cov(currentData);
end

maxIter = 1e3;
tolerance = 1e-6;
for tt=1:maxIter
    % E step, calculate alpha-i,t
    alpha = zeros(N,K);
    
    for ii=1:K
        for jj=1:N
            alpha(jj,ii) = GetLogMvnLikelihood(data(jj,:)',mu{ii},sigma{ii})+log(piParam(ii));
        end
    end
    
    for ii=1:N
        tmp = alpha(ii,:);
        tmp = tmp-LogSum(tmp,K);
        alpha(ii,:) = tmp; % log of alpha values, i.e. the probability 
                       % that a given datapoint comes from a given
                       % component, given values of that datapoint and the
                       % current estimate of mu and sigma for that
                       % component (under the multivariate Gaussian model)
    end
    
    % M step, calculate new values for parameters pi, mu, sigma
    alphaSum = zeros(K,1);
    for ii=1:K
        alphaSum(ii) = exp(LogSum(alpha(:,ii),N));
    end
    piParamStar = alphaSum/N;
    
    muStar = mu;
    sigmaStar = sigma;
    for ii=1:K
        for jj=1:d
           muStar{ii}(jj) = exp(LogSum(alpha(:,ii)+logData(:,jj),N)-LogSum(alpha(:,ii),N)); 
        end
        
        tmp = zeros(N,d,d);
        for jj=1:N
            tmp(jj,:,:) = log((data(jj,:)'-muStar{ii})*(data(jj,:)'-muStar{ii})');
        end
        
        for jj=1:d
            for kk=1:d
                sigmaStar{ii}(jj,kk) = exp(LogSum(alpha(:,ii)+squeeze(tmp(:,jj,kk)),N)-LogSum(alpha(:,ii),N));
            end
        end
    end
    
    totalPrecision = 0;
    for ii=1:K
       totalPrecision = totalPrecision+sum(abs(muStar{ii}-mu{ii}));
       totalPrecision = totalPrecision+sum(sum(abs(sigmaStar{ii}-sigma{ii})));
    end
    if totalPrecision<=tolerance
        % get back real numbers, in case original inputs were complex
        % numbers (this is the case if datapoints are negative)
        piParam = real(piParam);
        alpha = real(alpha);
        for ii=1:K
            tmp = mu{ii};
            mu{ii} = real(tmp);
            tmp = sigma{ii};
            sigma{ii} = real(tmp);
        end
        break;
    end
    piParam = piParamStar;
    mu = muStar;
    sigma = sigmaStar;
end
end

function [logPDF] = GetLogMvnLikelihood(data,mu,sigma)
logdet = 2*sum(log(diag(chol(sigma))));
logPDF = -0.5*logdet-0.5*(data-mu)'*(sigma\(data-mu));

end

function [X] = SWEEP(X,k)
%SWEEP.m
%   Code to evaluate the SWEEP operator on row k of matrix X
%Inputs: 
%        X - a matrix, n-by-p
%        k - the row number on which to evaluate the operator, k<=n
%
%Outputs: 
%        X - the matrix after performing the SWEEP operator on row k
%
% Created by: Byron Price
% 2018/10/08

D = X(k,k);
X(k,:) = X(k,:)./D;

[n,~] = size(X);
for ii=[1:k-1,k+1:n]
   B = X(ii,k);
   X(ii,:) = X(ii,:)-B.*X(k,:);
   X(ii,k) = -B/D;
end
X(k,k) = 1/D;
end

function [summation] = LogSum(vector,vectorLen)
if vectorLen==0
    summation = -Inf;
elseif vectorLen==1
    summation = vector(1);
else
    vector = sort(vector);
    summation = LogSumExpTwo(vector(1),vector(2));
    for ii=2:vectorLen-1
        summation = LogSumExpTwo(summation,vector(ii+1));
    end
end

end

function [y] = LogSumExpTwo(x1,x2)
check = x1>=x2;
if check==1
    y = x1+SoftPlus(x2-x1);
else
    y = x2+SoftPlus(x1-x2);
end
end

function [y] = SoftPlus(x)
if x<-34 % condition for small x
   y = 0;
else
   y = log(1+exp(-x))+x; % numerically stable calculation of log(1+exp(x))
end

end

