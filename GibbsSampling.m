% GibbsSampling.m
%  example of Gibbs sampling under a change-point model
%   from Bayesian Inference: Gibbs Sampling, Yildirim 2012

N = 250;
truen = 117;

lambda1 = 2;
lambda2 = 5;

fprintf('Change-Point Model\n');
fprintf('True Change Point n: %d\n',truen);
fprintf('True Lambda-1: %d\n',lambda1);
fprintf('True Lambda-2: %d\n\n',lambda2);

data = zeros(N,1);

data(1:truen) = poissrnd(lambda1,[truen,1]);
data(truen+1:N) = poissrnd(lambda2,[N-truen,1]);

a = 2;b=1/1; % the paper I reference uses the 
           % other parameterization of the gamma distribution
           % so his b=1 is my b-hat = 1/b

% RUN MCMC FIRST
tic;
numParams = 3;
numIter = 7e4;
burnIn = 1e4;
updateParam = 1e-2;
posterior = zeros(numIter,1);
allParams = zeros(numParams,numIter);

allParams(1,1) = random('Discrete Uniform',N);
allParams(2,1) = gamrnd(a,b);
allParams(3,1) = gamrnd(a,b);

lBound = [1,0,0]';
uBound = [N,Inf,Inf]';

mu = zeros(numParams,1);
sigma = eye(numParams);
sigma(1,1) = 200;
sigma(2,2) = 0.1;
sigma(3,3) = 0.1;

loglambda = log(2.38^2/numParams);
updateMu = mvnrnd(mu,sigma)';
optimalAccept = 0.44;%0.234;

ML = sum(log(poisspdf(data(1:round(allParams(1,1))),allParams(2,1))));
ML = ML+sum(log(poisspdf(data(round(allParams(1,1))+1:end),allParams(3,1))));

prior = log(gampdf(allParams(2,1),a,b))+log(gampdf(allParams(3,1),a,b))-log(N);

posterior(1) = ML+prior;

% scatter(1,posterior(1));hold on;pause(0.1);
for ii=2:numIter
   Z = mvnrnd(mu,exp(loglambda)*sigma)';
   pStar = allParams(:,ii-1)+Z;
   
   if sum(pStar<=lBound) == 0 && sum(pStar>=uBound) == 0
       changepoint = round(pStar(1));
       ML = sum(log(poisspdf(data(1:changepoint),pStar(2))));
       ML = ML+sum(log(poisspdf(data(changepoint+1:end),pStar(3))));
       
       prior = log(gampdf(pStar(2),a,b))+log(gampdf(pStar(3),a,b))-log(N);
       
       logA = (ML+prior)-posterior(ii-1);
       
       if log(rand) < logA
           allParams(:,ii) = pStar;
           posterior(ii) = ML+prior;
       else
           allParams(:,ii) = allParams(:,ii-1);
           posterior(ii) = posterior(ii-1);
       end
   else
       allParams(:,ii) = allParams(:,ii-1);
       posterior(ii) = posterior(ii-1);
       logA = -50;
   end
%     scatter(ii,posterior(ii));hold on;pause(0.01);
    sigma = sigma+updateParam.*((allParams(:,ii)-updateMu)*...
        (allParams(:,ii)-updateMu)'-sigma);
    updateMu = updateMu+updateParam.*(allParams(:,ii)-updateMu);
    loglambda = loglambda+updateParam.*(exp(min(0,logA))-optimalAccept);
end

[estMAP,ind] = max(posterior);
paramMAP = allParams(:,ind);

figure();stem(1:N,data);hold on;
n = round(paramMAP(1));
plot(1:n,ones(n,1).*paramMAP(2),'k','LineWidth',3);
plot(n+1:N,ones(N-n,1).*paramMAP(3),'k','LineWidth',3);
title('Change-Point Model MAP Fit');

posteriorMCMCSamples = allParams(:,burnIn:50:end);


figure();numRows = ceil(numParams/2);
for ii=1:numParams
   subplot(numRows,2,ii);histogram(posteriorMCMCSamples(ii,:));
   title('MCMC');
end

fprintf('MAP Parameter Estimates\n');
fprintf('Change Point n: %3.2f\n',paramMAP(1));
fprintf('Lambda-1: %3.2f\n',paramMAP(2));
fprintf('Lambda-2: %3.2f\n\n',paramMAP(3));
fprintf('MCMC Time\n');
toc;

% GIBBS SAMPLER
tic;
burnIn= 500;
numIter = 12500;
allParams = zeros(numParams,numIter);
allParams(1,1) = random('Discrete Uniform',N);
allParams(2,1) = gamrnd(a,b);
allParams(3,1) = gamrnd(a,b);

for ii=2:numIter
    n = allParams(1,ii-1);
    sum1 = sum(data(1:n));
    sum2 = sum(data(n+1:N));
    allParams(2,ii) = gamrnd(a+sum1,1/(n+b));
    allParams(3,ii) = gamrnd(a+sum2,1/(N-n+b));
    
    multinom = zeros(N,1);
    for jj=1:N
       sum1 = sum(data(1:jj));
       sum2 = sum(data(jj+1:N));
       multinom(jj) = sum1*log(allParams(2,ii))-jj*allParams(2,ii)+...
           sum2*log(allParams(3,ii))-(N-jj)*allParams(3,ii);
    end
    multinom = exp(multinom);
    cdf = cumsum(multinom./sum(multinom));
    randNum = rand;
    [~,ind] = min(abs(cdf-rand));
    allParams(1,ii) = ind;
end

posteriorGibbsSamples = allParams(:,burnIn:10:numIter);

figure();numRows = ceil(numParams/2);
for ii=1:numParams
   subplot(numRows,2,ii);histogram(posteriorGibbsSamples(ii,:));
   title('Gibbs');
end
fprintf('Gibbs Sampler Time\n');
toc;