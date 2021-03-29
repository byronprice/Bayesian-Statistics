function [piParam,mu,sigma,alpha,logLikelihood] = GaussMixtureEM(data,K)
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
%       logLikelihood - log likelihood of the data under the fitted model
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
sigmaInv = cell(K,1);
Id = eye(d);
for ii=1:K
   currentData = data(data(:,1)>=Q(1,ii) & data(:,1)<Q(1,ii+1),:);
   mu{ii} = mean(currentData)';
   sigmaInv{ii} = cov(currentData)\Id;
end

maxIter = 1e3;
tolerance = 1e-6;
prevLikelihood = -Inf;
for tt=1:maxIter
    % E step, calculate alpha-i,t
    alpha = zeros(N,K);
    
    for ii=1:K
        for jj=1:N
            alpha(jj,ii) = GetLogMvnLikelihood(data(jj,:)',mu{ii},sigmaInv{ii})+log(piParam(ii));
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
        alphaSum(ii) = LogSum(alpha(:,ii),N);
    end
    piParamStar = real(exp(alphaSum))/N;
    
    muStar = mu;
    sigmaStar = sigmaInv;
    for ii=1:K
        for jj=1:d
           muStar{ii}(jj) = real(exp(LogSum(alpha(:,ii)+logData(:,jj),N)-alphaSum(ii))); 
        end
        
        tmp = zeros(N,d,d);
        for jj=1:N
            tmp(jj,:,:) = log((data(jj,:)'-muStar{ii})*(data(jj,:)'-muStar{ii})')+...
                alpha(jj,ii);
        end
        
        for jj=1:d
            for kk=1:d
                sigmaStar{ii}(jj,kk) = real(exp(LogSum(squeeze(tmp(:,jj,kk)),N)-alphaSum(ii)));
            end
        end
        sigmaStar{ii} = sigmaStar{ii}\Id;
    end
    
    logLikelihood = GetLogLikelihood(data,mu,sigmaInv,piParam,N,K);
    logLikelihood-prevLikelihood
    if (logLikelihood-prevLikelihood)<=tolerance
        break;
    end
    piParam = piParamStar;
    mu = muStar;
    sigmaInv = sigmaStar;
    prevLikelihood = logLikelihood;
end

% evaluate log likelihood of the data

% logLikelihood = GetLogLikelihood(data,mu,sigma,piParam,N,K);
sigma = cell(K,1);
for kk=1:K
   sigma{kk} = sigmaInv{kk}\Id;
end
end

function [loglike] = GetLogLikelihood(data,mu,sigmaInv,piParam,N,K)
loglike = 0;
for nn=1:N
    summation = zeros(K,1);
    for kk=1:K
        summation(kk) = log(piParam(kk))+GetLogMvnLikelihood(data(nn,:)',mu{kk},sigmaInv{kk});
    end
    loglike = loglike+LogSum(summation,K);
end
end

function [logPDF] = GetLogMvnLikelihood(data,mu,sigmaInv)
logdet = sum(log(diag(chol(sigmaInv))));
logPDF = logdet-0.5*(data-mu)'*(sigmaInv*(data-mu));

end

function [summation] = LogSum(vector,vectorLen)
if vectorLen==0
    summation = -Inf;
elseif vectorLen==1
    summation = vector(1);
else

    maxVal = max(vector);
    difference = vector-maxVal;
    summation = maxVal+log1p(sum(exp(difference))-1);
    
    if isnan(summation) || summation>1e12
        vector = sort(vector);
        summation = LogSumExpTwo(vector(1),vector(2));
        for ii=2:vectorLen-1
            summation = LogSumExpTwo(summation,vector(ii+1));
        end
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
% if x<-34 % condition for small x
%    y = 0;
% else
%    y = log(1+exp(-x))+x; % numerically stable calculation of log(1+exp(x))
% end

y = log(1+exp(-x))+x;
y(x<=-34) = 0;

end

