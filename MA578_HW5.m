% MA578_HW5.m
%% PROBLEM 1 - bioassay data
X = [-0.86;-0.3;-0.05;0.73];
N = [5;5;5;5];
Y = [0;1;3;5];

K = 4;
[b,~,stats] = glmfit([ones(K,1),X],[Y,N],'binomial','constant','off');

numIter = 2e5;
burnIn = 1e5;

params = zeros(numIter,2);
params(1,:) = [0.85,7.75];

invlogitFun = @(a,b,x) 1./(1+exp(-(a+b.*x)));
loglikelihood = @(theta,y,n) Y.*log(theta)+(N-Y).*log(1-theta);

posterior = sum(loglikelihood(invlogitFun(params(1,1),params(1,2),X),Y,N));

for ii=2:numIter
    paramStar = params(1,:)+normrnd([0,0],[1,4.8]);
    
    posteriorStar = sum(loglikelihood(invlogitFun(paramStar(1),paramStar(2),X),Y,N));
    
    logA = posteriorStar-posterior;
    
    if log(rand)<logA
        params(ii,:) = paramStar;
        posterior = posteriorStar;
    else
        params(ii,:) = params(ii-1,:);
    end
%     plot(ii,posterior,'.');hold on;pause(1/100);
end
params = params(burnIn+1:100:end,:);
figure;subplot(2,1,1);histogram(params(:,1));
title('Marginal Posterior \alpha');
subplot(2,1,2);histogram(params(:,2));
title('Marginal Posterior \beta');

figure;hist3(params);

figure;plot(params(:,1),params(:,2),'.');
title('Posterior Samples');

alpha = 0.05;
quantile(params,[alpha/2,1-alpha/2])

%% PROBLEM 2 - models for machine quality control
%   data
Y = [83,92,92,46,67;117,109,114,104,87;101,93,92,86,67;105,119,116,102,116;...
    79,97,103,79,92;57,92,104,77,100];


%   MODEL 1 - SEPARATE
N = [5,5,5,5,5,5];
nJ = 6;

numIter = 2e5;burnIn = 1e5;

params = zeros(numIter,nJ*2);

suffStats = zeros(nJ,1);
count = 1;
for jj=1:nJ
    suffStats(jj) = mean(Y(jj,:));
    params(1,count) = suffStats(jj);
    count = count+1;
    params(1,count) = sum((Y(jj,:)-params(1,count-1)).^2)./N(jj);
    count = count+1;
end
for ii=2:numIter
    count = 1;
    for jj=1:nJ
        params(ii,count) = normrnd(suffStats(jj),sqrt(params(ii-1,count+1)./N(jj)));
        count = count+1;
        params(ii,count) = 1/gamrnd(N(jj)/2,1/(sum((Y(jj,:)-params(ii,count-1)).^2)/2));
        count = count+1;
    end
end
params = params(burnIn+1:100:end,:);

alpha = 0.05;
figure;
count = 1;
for jj=1:nJ*2
   if mod(count,2)==1
      subplot(nJ,2,count);histogram(params(:,count));
      axis([50 150 0 200]);
      set(gca,'YTickLabel',[]);
      t = title(sprintf('Machine %d : \\mu',round(jj/2)));
      t.FontSize = 16;
      quantile(params(:,count),[alpha/2,1-alpha/2])
   else
       subplot(nJ,2,count);histogram(sqrt(params(:,count)));
       axis([0 50 0 200]);
       set(gca,'YTickLabel',[]);
       t = title(sprintf('Machine %d : \\sigma',round(jj/2)));
       t. FontSize = 16;
       quantile(params(:,count),[alpha/2,1-alpha/2])
   end
   count = count+1;
end

% predictive for machine 6
predictive = zeros(1000,1);
for ii=1:1000
    mu = params(ii,11);
    sigma = sqrt(params(ii,12));
    predictive(ii) = normrnd(mu,sigma);
end

quantile(predictive,[alpha/2,1-alpha/2])

%% MODEL 2 - POOLED
params = zeros(numIter,2);totalN = sum(N);
suffStats = [mean(Y(:)),var(Y(:))];
params(1,:) = suffStats;
for ii=2:numIter
    params(ii,1) = normrnd(suffStats(1),sqrt(params(ii-1,2)/totalN));
    params(ii,2) = 1/gamrnd(totalN/2,1/(sum((Y(:)-params(ii,1)).^2)/2));
end
params = params(burnIn+1:100:end,:);

alpha = 0.05;
figure;
count = 1;
for jj=1:nJ
      subplot(nJ,2,count);histogram(params(:,1));
      axis([50 150 0 200]);
      set(gca,'YTickLabel',[]);
      t = title(sprintf('Machine %d : \\mu',round(count/2)));
      t.FontSize = 16;
      quantile(params(:,1),[alpha/2,1-alpha/2])
      count = count+1;
      
      subplot(nJ,2,count);histogram(sqrt(params(:,2)));
      axis([0 50 0 200]);
      set(gca,'YTickLabel',[]);
      t = title(sprintf('Machine %d : \\sigma',round(count/2)));
      t. FontSize = 16;
      quantile(params(:,2),[alpha/2,1-alpha/2])
      count = count+1;
end

% predictive for machine 6
predictive = zeros(1000,1);
for ii=1:1000
    mu = params(ii,1);
    sigma = sqrt(params(ii,2));
    predictive(ii) = normrnd(mu,sigma);
end

quantile(predictive,[alpha/2,1-alpha/2])

%% MODEL 3 - HIERARCHICAL

numIter = 2e5;burnIn = 1e5;

params = zeros(numIter,nJ+3);

suffStats = zeros(nJ,1);
sigSquareSum = 0;
for jj=1:nJ
    suffStats(jj) = mean(Y(jj,:));
    params(1,jj) = suffStats(jj);
    
    sigSquareSum = sigSquareSum+sum((Y(jj,:)-params(1,jj)).^2);
end
params(1,nJ+1) = 1/gamrnd(totalN/2,1/(sigSquareSum/2));
params(1,nJ+2) = normrnd(mean(params(1,1:nJ)),sqrt(var(Y(:))/nJ));
params(1,nJ+3) = 1/gamrnd((nJ-1)/2,1/(sum((params(1,1:nJ)-params(1,nJ+2)).^2)/2));

for ii=2:numIter
    sigSquareSum = 0;
    for jj=1:nJ
        sigsquare = params(ii-1,nJ+1);
        mu = params(ii-1,nJ+2);
        tausquare = params(ii-1,nJ+3);
        
        denom = N(jj)/sigsquare+1/tausquare;
        numer = suffStats(jj)*N(jj)/sigsquare+mu/tausquare;
        
        params(ii,jj) = normrnd(numer/denom,sqrt(1/denom));
        
        sigSquareSum = sigSquareSum+sum((Y(jj,:)-params(ii,jj)).^2);
    end
    params(ii,nJ+1) = 1/gamrnd(totalN/2,1/(sigSquareSum/2));
    params(ii,nJ+2) = normrnd(mean(params(ii,1:nJ)),...
        sqrt(params(ii-1,nJ+3)/nJ));
    params(ii,nJ+3) = 1/gamrnd((nJ-1)/2,1/(sum((params(ii,1:nJ)-params(ii,nJ+2)).^2)/2));
end
params = params(burnIn+1:100:end,:);

alpha = 0.05;
figure;
count = 1;
for jj=1:nJ
      subplot(nJ,2,count);histogram(params(:,jj));
      axis([50 150 0 200]);
      set(gca,'YTickLabel',[]);
      t = title(sprintf('Machine %d : \\mu',round(count/2)));
      t.FontSize = 16;
      quantile(params(:,jj),[alpha/2,1-alpha/2])
      count = count+1;
      
      subplot(nJ,2,count);histogram(sqrt(params(:,nJ+1)));
      axis([0 50 0 200]);
      set(gca,'YTickLabel',[]);
      t = title(sprintf('Machine %d : \\sigma',round(count/2)));
      t. FontSize = 16;
      
      count = count+1;
end
quantile(params(:,nJ+1),[alpha/2,1-alpha/2])
% predictive for machine 6
predictive = zeros(1000,1);
for ii=1:1000
    mu = params(ii,6);
    sigma = sqrt(params(ii,nJ+1));
    predictive(ii) = normrnd(mu,sigma);
end

quantile(predictive,[alpha/2,1-alpha/2])

% posterior for 7
posterior = zeros(1000,1);
for ii=1:1000
    mu = params(ii,nJ+2);
    tau = sqrt(params(ii,nJ+3));
    posterior(ii) = normrnd(mu,tau);
end
figure;histogram(posterior);
quantile(posterior,[alpha/2,1-alpha/2])
%% PROBLEM 3 - survival time data on patients with multiple
%  myeloma

y = [13,52,6,40,10,7,66,10,10,14,16,4,65,5,...
    11,10,15,5,76,56,88,24,51,4,40,8,18,5,16,50,...
    40,1,36,5,10,91,18,1,18,6,1,23,15,18,12,17,3];

ySuffStat = sum(log(y));

n = length(y);

lambdaInit = 0.01;
w = BoxCox(y,lambdaInit);
wmean = mean(w);
sigsquare = var(w);

numIter = 3e5;
params = zeros(2,numIter,3);
posterior = zeros(numIter,2);

params(1,1,:) = [lambdaInit,wmean,sigsquare];
params(2,1,:) = [lambdaInit,wmean,sigsquare];

posterior(1,:) = (-1./(2*params(1,1,3)))*sum((w-params(1,1,2)).^2)+(params(1,1,1)-1).*ySuffStat;

loglambda = -5.5;optimalAccept = 0.434;
for ii=2:numIter
    for jj=1:2
        w = BoxCox(y,params(jj,ii-1,1));
        wmean = mean(w);
        
        % gibbs steps
        params(jj,ii,2) = normrnd(wmean,sqrt(params(jj,ii-1,3)/n));
        
        suffStat = sum((w-params(jj,ii,2)).^2);%sum((w-wmean).^2)+n*(wmean-params(ii,2)).^2;
        params(jj,ii,3) = 1/gamrnd((n-1)/2,1/(suffStat/2));
        
        
        % metropolis-hastings step
        checkPosterior = (-1/(2*params(jj,ii,3)))*sum((w-params(jj,ii,2)).^2)+(params(jj,ii-1,1)-1).*ySuffStat;
        
        lambdaStar = params(jj,ii-1,1)+normrnd(0,sqrt(exp(loglambda)));
        
        w = BoxCox(y,lambdaStar);
        posteriorStar = (-1/(2*params(jj,ii,3)))*sum((w-params(jj,ii,2)).^2)+(lambdaStar-1).*ySuffStat;
        logA = posteriorStar-checkPosterior;
        
        if log(rand)<logA
            params(jj,ii,1) = lambdaStar;
            posterior(ii,jj) = posteriorStar;
        else
            params(jj,ii,1) = params(jj,ii-1,1);
            posterior(ii,jj) = checkPosterior;
        end
    end
%     loglambda = loglambda+0.01*(exp(min(0,logA))-optimalAccept);
    
%     plot(ii,posterior(ii),'.');hold on;pause(1/100);
end

burnIn = 1e5;

params = params(:,burnIn+1:200:end,:);
posterior = posterior(burnIn+1:200:end,:);

figure;subplot(3,1,1);plot(params(1,:,1));hold on;plot(params(2,:,1));
title('Trace Plot: \lambda');
subplot(3,1,2);plot(params(1,:,2));hold on;plot(params(2,:,2));
title('Trace Plot: \mu');
subplot(3,1,3);plot(params(1,:,3));hold on;plot(params(2,:,3));
title('Trace Plot: \sigma^2');


figure;subplot(3,1,1);autocorr(params(1,:,1));
title('Autocorrelation: \lambda');
subplot(3,1,2);autocorr(params(1,:,2));
title('Autocorrelation: \mu');
subplot(3,1,3);autocorr(params(1,:,3));
title('Autocorrelation: \sigma^2');

figure;subplot(2,1,1);plot(posterior(:,1));hold on;plot(posterior(:,2));
title('Trace Plot: Log Conditional Posterior \lambda');
subplot(2,1,2);autocorr(posterior(:,1));
title('Autocorr: Log Conditional Posterior \lambda');

alpha = 0.05;

for ii=1:3
    quantile(params(1,:,ii),[alpha/2,1-alpha/2])
end

% posterior predictive
ytilde = zeros(length(params(1,:,1)),1);
for ii=1:length(params(1,:,1))
    mu = params(1,ii,2);sigma = sqrt(params(1,ii,3));
    
    w = normrnd(mu,sigma);
    ytilde(ii) = InvBoxCox(w,params(1,ii,1));
    
    while ~isreal(ytilde(ii))
        w = normrnd(mu,sigma);
        ytilde(ii) = InvBoxCox(w,params(1,ii,1));
    end
end

figure;histogram(y,'Normalization','Probability');hold on;
title('Posterior Predictive Check');
histogram(ytilde,'Normalization','Probability');
legend('Data','Predictive');
quantile(ytilde,[alpha/2,1-alpha/2])

x = [y';ytilde];
g = [repmat({'Data'},n,1);repmat({'Predictive'},length(params(1,:,1)),1)];
figure;boxplot(x,g);
title('Posterior Predictive Check');
