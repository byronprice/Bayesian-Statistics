function [params,posterior,decision,hpdInt,latentVar] = BayesTtest(data1,data2,alpha)
% BEST.m
%  Bayesian estimation from "Bayesian Estimation Supersedes the t-test"
%   paper by Kruschke 2013
%  Bayesian credible interval and highest posterior density interval from 
%    "Monte Carlo Estimation of Bayesian Credible and HPD Intervals"
%     Author(s): Ming-Hui Chen and Qi-Man Shao 1999
%  Takes data from 2 groups and compares them, comparable to a t-test
%  Able to make moderate adjustments for different data likelihoods and
%   hierarchical models

%  Beta-Bernoulli hierarchical model ... the real question, are these two
%   distributions or one? So, we have a beta prior on a bernoulli random
%   variable which differentiates between 1 distribution (0) or two
%   distributions (1), using a point mass mixture prior

if nargin<2
    disp('Must enter two datasets');
    return;
elseif nargin<3
    alpha = 0.05;
end

data1 = data1(:);data2 = data2(:);

% check data for autocorrelations
N1 = length(data1);
tmp1 = autocorr(data1,min(round(N1/2),50));
ess1 = 1/(1+2*sum(tmp1(2:end)));
N2 = length(data2);
tmp2 = autocorr(data2,min(round(N2/4),50));
ess2 = 1/(1+2*sum(tmp2(2:end)));

poolData = [data1;data2];
poolMean = mean(poolData);rope = max(0.01,abs(poolMean)/100);
poolStdev = std(poolData);

burnIn = 1e4;
numSamples = 5e3; % 2000
numSkip = 100;% 50
numIter = numSamples*numSkip+burnIn;

numParams = 7;
params = zeros(numIter,numParams);
posterior = zeros(numIter,1);

prior = @tPrior;likelihood = @tLikelihood;
iter = 1;
params(iter,:) = [log(0.5),0,0,0,...
    mean(data1(:)),log(std(data1(:))),log(10)];

posterior(iter) = pointPrior(params(iter,1:4),rope)+betaPrior(params(iter,1));

group2Mean = params(iter,5)+params(iter,2);
group2Std = params(iter,6)+params(iter,3);
group2Nu = params(iter,7)+params(iter,4);
posterior(iter) = posterior(iter)+prior([group2Std,params(iter,6)],[group2Nu,params(iter,7)])+...
    likelihood(data1,[params(iter,5),params(iter,6),params(iter,7)])+...
    likelihood(data2,[group2Mean,group2Std,group2Nu]);

prevNonZeroDiff = [0,0,0];

latentVar = zeros(numIter,1);latentVar(iter) = 0;

jumpInds = [1,5,6,7];numJump = 4;
updateParam = 0.01;
loglambda = ones(1,numParams);
loglambda(1) = -5;loglambda(2) = log(abs(poolMean)/10);
loglambda(3) = log(abs(poolStdev)/10);
loglambda(4) = -5;
loglambda(5) = log(abs(params(iter,5))/10);
loglambda(6) = log(abs(params(iter,6))/10);
loglambda(7) = -5;
optimalAccept = 0.234;
for iter=2:burnIn
    % add some kind of check so that we only change to indices that are
    %  plausible

    params(iter,:) = params(iter-1,:);
    index = jumpInds(unidrnd(numJump));
    if index==1
        params(iter,index) = log(betarnd(0.5,0.5));
    else
        stdev = sqrt(exp(loglambda(index)));
        params(iter,index) = params(iter,index)+normrnd(0,stdev);
    end

    latentVar(iter) = log(rand)<params(iter,1); %-SoftPlus(-params(iter,1));
    if latentVar(iter) == 0
        params(iter,2:4) = 0;
    else
        params(iter,2) = prevNonZeroDiff(1)+normrnd(0,sqrt(exp(loglambda(2))));
        params(iter,3) = prevNonZeroDiff(2)+normrnd(0,sqrt(exp(loglambda(3))));
        params(iter,4) = prevNonZeroDiff(3)+normrnd(0,sqrt(exp(loglambda(4))));
    end
    
    posterior(iter) = pointPrior(params(iter,1:4),rope)+betaPrior(params(iter,1));
    
    group2Mean = params(iter,5)+params(iter,2);
    group2Std = params(iter,6)+params(iter,3);
    group2Nu = params(iter,7)+params(iter,4);
    
    if latentVar(iter)>-1 %==1
        posterior(iter) = posterior(iter)+prior([group2Std,params(iter,6)],[group2Nu,params(iter,7)])+...
            likelihood(data1,[params(iter,5),params(iter,6),params(iter,7)])+...
            likelihood(data2,[group2Mean,group2Std,group2Nu]);
    else
        posterior(iter) = posterior(iter)+prior(params(iter,6),params(iter,7))+...
            likelihood(data1,[params(iter,5),params(iter,6),params(iter,7)])+...
            likelihood(data2,[group2Mean,group2Std,group2Nu]);
    end

    logA = posterior(iter)-posterior(iter-1);
    
    if log(rand) < logA
        % keep current params
    else
        % revert to previous
        params(iter,:) = params(iter-1,:);
        posterior(iter) = posterior(iter-1);
        latentVar(iter) = latentVar(iter-1);
    end
    
    if latentVar(iter)==1
        loglambda(index) = loglambda(index)+updateParam.*(exp(min(0,logA))-optimalAccept);
        loglambda(2:4) = loglambda(2:4)+updateParam.*(exp(min(0,logA))-optimalAccept);
        
        prevNonZeroDiff = params(iter,2:4);
    else
        loglambda(index) = loglambda(index)+updateParam.*(exp(min(0,logA))-optimalAccept);
    end
    
%     exp(-SoftPlus(-params(iter,2)))
%     plot(iter,-SoftPlus(-params(iter,2)),'.');hold on;pause(1/100);
end

proposal = sqrt(exp(loglambda))
proposal(1) = 1;
for iter=burnIn+1:numIter
    params(iter,:) = params(iter-1,:);
    index = jumpInds(unidrnd(numJump));
    if index==1
        params(iter,index) = log(betarnd(0.5,0.5));
    else
        stdev = sqrt(exp(loglambda(index)));
        params(iter,index) = params(iter,index)+normrnd(0,stdev);
    end

    latentVar(iter) = log(rand)<params(iter,1); %-SoftPlus(-params(iter,1));
    if latentVar(iter) == 0
        params(iter,2:4) = 0;
    else
        params(iter,2) = prevNonZeroDiff(1)+normrnd(0,proposal(2));
        params(iter,3) = prevNonZeroDiff(2)+normrnd(0,proposal(3));
        params(iter,4) = prevNonZeroDiff(3)+normrnd(0,proposal(4));
    end
    
    posterior(iter) = pointPrior(params(iter,1:4),rope)+betaPrior(params(iter,1));
    
    group2Mean = params(iter,5)+params(iter,2);
    group2Std = params(iter,6)+params(iter,3);
    group2Nu = params(iter,7)+params(iter,4);
    
    if latentVar(iter)>-1 %==1
        posterior(iter) = posterior(iter)+prior([group2Std,params(iter,6)],[group2Nu,params(iter,7)])+...
            likelihood(data1,[params(iter,5),params(iter,6),params(iter,7)])+...
            likelihood(data2,[group2Mean,group2Std,group2Nu]);
    else
        posterior(iter) = posterior(iter)+prior(params(iter,6),params(iter,7))+...
            likelihood(data1,[params(iter,5),params(iter,6),params(iter,7)])+...
            likelihood(data2,[group2Mean,group2Std,group2Nu]);
    end
    
    logA = posterior(iter)-posterior(iter-1);
    
    if log(rand) < logA
        % keep current params
    else
        % revert to previous
        params(iter,:) = params(iter-1,:);
        posterior(iter) = posterior(iter-1);
        latentVar(iter) = latentVar(iter-1);
    end
    
    if latentVar(iter)==1
       prevNonZeroDiff = params(iter,2:4); 
    end
end

params = params(burnIn+1:numSkip:end,:);
posterior = posterior(burnIn+1:numSkip:end);
latentVar = latentVar(burnIn+1:numSkip:end);

% params(:,1) = -SoftPlus(-params(:,1));
params(:,3) = exp(params(:,3)+params(:,6));
params(:,4) = SoftPlus(params(:,4)+params(:,7));
params(:,6) = exp(params(:,6));
params(:,7) = SoftPlus(params(:,7));
numSamples = size(params,1);

% make a decision based on the highest posterior density interval
%  of the beta-prior variable
sortedSamples = sort(params(:,1));
numInds = ceil(numSamples*(1-alpha));

hpdInt = quantile(sortedSamples,[alpha/2,1-alpha/2]); % Bayesian credible interval
width = hpdInt(2)-hpdInt(1);
for ii=1:numSamples-numInds+1
    tempInt = [sortedSamples(ii),sortedSamples(ii+numInds-1)];
    if (tempInt(2)-tempInt(1))<width
        width = tempInt(2)-tempInt(1);
        hpdInt = tempInt;
    end
end

if min(hpdInt)>log(1-alpha)
    decision = 1;
elseif max(hpdInt)<log(alpha)
    decision = -1;
else
    decision = 0;
end
% decision = min(hpdInt)>log((1-alpha));

hpdInt = exp(hpdInt);
end

function [loglikelihood] = tLikelihood(data,parameters)
mu = parameters(1);sigma = parameters(2);nu = parameters(3);

constant = gammaln(0.5*((exp(nu)+1)+1))-...
    sigma-0.5*SoftPlus(nu)-gammaln(0.5*(exp(nu)+1));
loglikelihood = constant+log(1+((data(:)-mu)./exp(sigma)).^2./(exp(nu)+1)).*(-0.5*((exp(nu)+1)+1));

loglikelihood = sum(loglikelihood);
end

function [prior] = tPrior(sigmaParams,nuParams)

lambda = 30;%sigsquare = rope*1e8;
% nuPrior = (1/lambda)*exp(-((exp(nu)+1)-1)/lambda); % nu >= 1

% sigmaPrior = 1/exp(sigma);
% muPrior = 1;
% epsilon = 1/1000;

prior = 0;
for ii=1:length(nuParams)
    prior = prior-log(1/lambda)-((exp(nuParams(ii))+1)-1)/lambda-...
        sigmaParams(ii);%-...
       % -0.5*log(sigsquare)-0.5*(muParams(ii)-poolMean).^2/sigsquare;
end

end

function [loglikelihood] = logNormalLikelihood(data,parameters)
mu = parameters(1);logsigma = parameters(2);n = size(data,1);
% consider getting sufficient stats
sigmasquare = exp(logsigma).^2;
loglikelihood = -n*logsigma-sum(log(data))-0.5*sum(log(data).^2)/sigmasquare+...
    sum(log(data).*mu)/sigmasquare-0.5*n*mu*mu/sigmasquare;
end

function [prior] = logNormalPrior(parameters)
mu = parameters(1);sigma = parameters(2);

% uniform on mu, 1/exp(sigma) on sigma
prior = -sigma;
end

function [loglikelihood] = GammaLikelihood(data,parameters)
a = parameters(1);b = parameters(2);n = size(data,1);
% consider getting sufficient statistics
loglikelihood = (a-1)*sum(log(data))-n*log(gamma(a))+n*a*log(b)-sum(data)*b;
end

function [prior] = GammaPrior(parameters)
loga = parameters(1);logb = parameters(2);

prior = -logb+0.5*log(exp(loga)*psi(1,exp(loga))-1)-0.5*loga;
end

function [prior] = pointPrior(parameters,rope)
x = parameters(2);logprob = parameters(1);%-SoftPlus(-parameters(1));% prior on difference of means

% point mass mixture prior
sigsquare = rope*1e8;
prior = log(-expm1(logprob))*(x==0)+(x~=0)*(logprob);%-0.5*log(sigsquare)-0.5*x.^2/sigsquare);
end

function [prior] = betaPrior(parameters)
x = parameters(1);%-SoftPlus(-parameters(1));
a = 0.5;b = 0.5;

logOneMinusX = log(-expm1(x));
prior = (a-1)*x+(b-1)*logOneMinusX;
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