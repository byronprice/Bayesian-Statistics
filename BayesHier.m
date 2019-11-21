function [params] = BayesHier(data,numSamples)
% BayesHier.m
%  Gibbs Sampler for Bayesian hierarchical Gaussian model
%    data is divided across groups (e.g. schools), j, and within each group
%    there are individual datapoints (e.g. students),i, each datapoint is
%    modeled as N(y-i,j | theta-j, sigma-squared) ... that is, each group has a
%    unique group mean but shared variance sigma-squared
%    each group mean is modeled as N(theta-j | mu, tau-squared) ... so,
%    each group mean itself comes from a shared normal distribution with
%    mean mu and variance tau-squared
%
%INPUT:
%      data - J-by-1 cell array, where J is the number of groups, within
%         each cell, there should be a vector of size nj-by-1, where nj is
%         the number of data points for group j
%      numSamples - number of posterior samples to output, defaults to 1000
%OUTPUT:
%      params - posterior samples for the J group means, the shared variance, 
%        and the hierarchical mean and variance
%
% (note that the code calculates the precision parameters, 1/sigma-squared
% and 1/tau-squared, and then converts back at the end)
%
%Created: 2019/11/07
%  Byron Price
%Updated: 2019/11/07
% By: Byron Price

if nargin<2
    numSamples = 1000;
end

J = length(data);
N = zeros(J,1);
for jj=1:J
   N(jj) = length(data{jj}); 
end

totalN = sum(N);

burnIn = 1e5;skipRate = 10;
numIter = burnIn+skipRate*numSamples;

params = zeros(numIter,J+3);

suffStats = zeros(J,1);
sigSquareSum = 0;
for jj=1:J
    suffStats(jj) = mean(data{jj});
    params(1,jj) = suffStats(jj);
    
    sigSquareSum = sigSquareSum+sum((data{jj}-params(1,jj)).^2);
end
params(1,J+1) = gamrnd(totalN/2,1/(sigSquareSum/2));
params(1,J+2) = normrnd(mean(params(1,1:J)),sqrt(var(data{1})/J));
params(1,J+3) = gamrnd((J-1)/2,1/(sum((params(1,1:J)-params(1,J+2)).^2)/2));

for ii=2:numIter
    sigsquare = params(ii-1,J+1);
    mu = params(ii-1,J+2);
    tausquare = params(ii-1,J+3);
        
    sigSquareSum = 0;
    for jj=1:J
        denom = N(jj)*sigsquare+tausquare;
        numer = suffStats(jj)*N(jj)*sigsquare+mu*tausquare;
        
        params(ii,jj) = normrnd(numer/denom,sqrt(1/denom));
        
        sigSquareSum = sigSquareSum+sum((data{jj}-params(ii,jj)).^2);
    end
    
    params(ii,J+1) = gamrnd(totalN/2,1/(sigSquareSum/2));
    params(ii,J+2) = normrnd(mean(params(ii,1:J)),...
        sqrt((1/params(ii-1,J+3))/J));
    params(ii,J+3) = gamrnd((J-1)/2,1/(sum((params(ii,1:J)-params(ii,J+2)).^2)/2));
end
params = params(burnIn+1:skipRate:end,:);

params(:,J+1) = 1./params(:,J+1);
params(:,J+3) = 1./params(:,J+3);

% do a quick check for convergence
ess = zeros(J+3,1);
for jj=1:J+3
    rho = autocorr(params(:,jj),floor(numSamples/2));
    ess(jj) = numSamples/(1+2*sum(rho));
    
    if ess/numSamples<0.3
       fprintf('Warning: Low effective sample size, which may indicate\n');
       fprintf('the chain did not reach convergence. Consider increasing the\n');
       fprintf('number of samples or the skip rate.\n');
    end
end

end