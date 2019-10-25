% MA578_HW4.m

% BDA problem 4.1 and 2.11
Y = [-2,-1,0,1.5,2.5];a = -4;b = 4;

cauchyFun = @(y,theta) 1./(1+(y-theta).^2);

priorFun = @(theta,a,b) theta>=a & theta<=b; 

thetagrid = linspace(-4,4,1e4);

logposterior = zeros(size(thetagrid));

for ii=1:length(thetagrid)
   logposterior(ii) = sum(log(cauchyFun(Y,thetagrid(ii)))).*priorFun(thetagrid(ii),a,b);
end

logsum = LogSum(logposterior,length(logposterior));

posterior = exp(logposterior-logsum);
plot(thetagrid,posterior);title('Cauchy Likelihood Posterior');
xlabel('\Theta');ylabel('PDF');

hold on;

% solve numerically for the mode
firstDeriv = zeros(size(thetagrid));
for ii=1:length(thetagrid)
   firstDeriv(ii) = sum(2.*(Y-thetagrid(ii))./(1+(Y-thetagrid(ii)).^2)); 
end
    
[~,ind] = min(abs(firstDeriv-0));

posteriorMode = thetagrid(ind);

% solve numerically for the information, - second deriv at mode
secondDeriv = sum(-2./(1+(Y-posteriorMode).^2)+(4.*(Y-posteriorMode).^2)./((1+(Y-posteriorMode).^2).^2));

information = -secondDeriv;

tmp = normpdf(thetagrid,posteriorMode,sqrt(1/information));
tmp = tmp./sum(tmp);

plot(thetagrid,tmp);legend('True Posterior','Normal Approx');


% problem 5.13
Y = [16,9,10,13,19,20,18,17,35,55]';J = length(Y);
X = [58,90,48,57,103,57,86,112,273,64]';
N = X+Y;

sigmoid = @(x) 1./(1+exp(-x));

logbinopdf = @(y,n,theta) y.*log(sigmoid(theta))+(n-y).*log(1-sigmoid(theta));
logbetapdf = @(theta,alpha,beta) gammaln(exp(alpha)+exp(beta))-gammaln(exp(alpha))-gammaln(exp(beta))...
    +(exp(alpha)-1).*log(sigmoid(theta))+(exp(beta)-1).*log(1-sigmoid(theta));

priorParam = -5/2;
alphabetaprior = @(alpha,beta) priorParam.*log(exp(alpha)+exp(beta));

numIter = 3e6;numParams = J+2;
posteriorSamples = zeros(numParams,numIter);

for jj=1:J
    p = Y(jj)/N(jj);
    posteriorSamples(jj,1) = log(p./(1-p));
end

posteriorSamples(J+1,1) = 1;
posteriorSamples(J+2,1) = 1;

logposterior = sum(logbinopdf(Y,N,posteriorSamples(1:J,1)))+...
    sum(logbetapdf(posteriorSamples(1:J,1),posteriorSamples(J+1,1),posteriorSamples(J+2,1)))+...
    alphabetaprior(posteriorSamples(J+1,1),posteriorSamples(J+2,1));

loglambda = ones(numParams,1).*log(2.38^2);
optimalAccept = 0.234;
updateParam = 1/100;
% figure;
for ii=2:numIter
    lambda = sqrt(exp(loglambda));
    index = randperm(numParams,1);
    
    pStar = posteriorSamples(:,ii-1);
    pStar(index) = pStar(index)+normrnd(0,lambda(index));
    
    
    newposterior = sum(logbinopdf(Y,N,pStar(1:J)))+...
        sum(logbetapdf(pStar(1:J),pStar(J+1),pStar(J+2)))+...
        alphabetaprior(pStar(J+1),pStar(J+2));
    
    logA = newposterior-logposterior;
    
    if log(rand)<logA
        posteriorSamples(:,ii) = pStar;
        logposterior = newposterior;
    else
        posteriorSamples(:,ii) = posteriorSamples(:,ii-1);
    end
    
    loglambda(index) = loglambda(index)+updateParam.*(exp(min(0,logA))-optimalAccept);
    

%     plot(ii,logposterior,'.');pause(1/100);hold on;
end

burnIn = 1e6;

posteriorSamples = posteriorSamples(:,burnIn+1:2000:end);

figure;hist3(posteriorSamples([J+1,J+2],:)');
xlabel('log(\alpha)');ylabel('log(\beta)');title('Marginal Posterior Samples');

figure;histogram(exp(posteriorSamples(J+1,:))./(exp(posteriorSamples(J+1,:))+exp(posteriorSamples(J+2,:))));
xlabel('Expectation of \theta');ylabel('Count');title('Marginal Posterior: Expectation of \theta');

figure;
for jj=1:J
   subplot(4,3,jj);histogram(sigmoid(posteriorSamples(jj,:)));
   axis([0 0.75 0 200]);hold on;plot(ones(10,1).*(Y(jj)./N(jj)),linspace(0,200,10),'k');
   xlabel(sprintf('\\theta-%d',jj));
end

% posterior interval for theta and predictive interval
newBlock = zeros(1000,1);theta = zeros(1000,1);
N = 100;
for ii=1:1000
alpha = exp(posteriorSamples(J+1,ii));beta = exp(posteriorSamples(J+2,ii));
theta(ii) = betarnd(alpha,beta);
y = binornd(N,theta(ii));
newBlock(ii) = y;
end
figure;histogram(newBlock);title('Posterior Predictive for N=100 Vehicles');
ylabel('Count');xlabel('Number of Bicycles');

quantile(theta,[0.05/2,1-0.05/2])
quantile(newBlock,[0.05/2,1-0.05/2])