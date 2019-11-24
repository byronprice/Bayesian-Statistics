% MA578_HW6.m

%% PROBLEM 1: BDA3 14.1
J = 3;
radonData = cell(J,3);
radonData{1,3} = 'BlueEarth';
radonData{2,3} = 'Clay';
radonData{3,3} = 'Goodhue';

radonData{1,1} = log([5,13,7.2,6.8,12.8,5.8,9.5,6,3.8,14.3,1.8,6.9,4.7,9.5]);
radonData{1,2} = [0,0,0,0,0,1,0,0,0,1,0,0,0,0];

radonData{2,1} = log([0.9,12.9,2.6,3.5,26.6,1.5,13,8.8,19.5,2.5,9,13.1,3.6,6.9]);
radonData{2,2} = [1,0,0,1,0,0,0,0,0,1,0,0,0,1];

radonData{3,1} = log([14.3,6.9,7.6,9.8,2.6,43.5,4.9,3.5,4.8,5.6,3.5,3.9,6.7]);
radonData{3,2} = [0,1,0,1,0,0,0,0,0,0,0,0,0];

% create design matrix
N = 0;
for jj=1:J
    N = N+length(radonData{jj,1});
end

K = 4;

X = zeros(N,K);Y = zeros(N,1);
count = 1;
for jj=1:J
    Y(count:count+length(radonData{jj,1})-1) = radonData{jj,1};
    X(count:count+length(radonData{jj,1})-1,jj) = 1;
    
    X(count:count+length(radonData{jj,1})-1,K) = radonData{jj,2};
    
    count = count+length(radonData{jj,1});
end

% get ML estimates
[betaHat,Vb] = MLestimate(X,Y);

% simulate from posterior
iter = 1000;
posteriorSamples = zeros(iter,K+1);

ssquare = (Y-X*betaHat)'*(Y-X*betaHat);
for ii=1:iter
    % simulate from inverse chi square
    sigsquare = 1/gamrnd((N-K)/2,2/ssquare);
    posteriorSamples(ii,1) = sigsquare;
    
    beta = mvnrnd(betaHat,Vb.*sigsquare);
    
    posteriorSamples(ii,2:end) = beta;
end

names = {'\sigma^2',radonData{1,3},radonData{2,3},radonData{3,3},'First-Floor'};
for ii=1:K+1
    subplot(3,2,ii);histogram(posteriorSamples(:,ii));
    title(names{ii});
end

% posterior summary
alpha = 0.05;
for ii=1:K+1
    tmp = posteriorSamples(:,ii);
    if ii==1 || ii==K+1
        quantile(tmp,[alpha/2,1-alpha/2])
    else
        quantile(exp(tmp),[alpha/2,1-alpha/2])
    end
end

% posterior predictive
predictive = zeros(iter,J,2);
figure;count = 1;
for jj=1:J
    countyInd = jj+1;
    for ii=1:2
        if ii==1
            firstfloor = 0;
        else
            firstfloor = 1;
        end
        for tt=1:iter
            sigsquare = posteriorSamples(tt,1);
            meanVal = posteriorSamples(tt,countyInd);
            
            if firstfloor==1
                meanVal = meanVal+posteriorSamples(tt,5);
            end
            newData = exp(normrnd(meanVal,sqrt(sigsquare)));
            
            predictive(tt,jj,ii) = newData;
        end
        subplot(3,2,count);histogram(squeeze(predictive(:,jj,ii)));
        count = count+1;
        if firstfloor==0
            title(sprintf('Basement %s',radonData{jj,3}));
        elseif firstfloor==1
            title(sprintf('First Floor %s',radonData{jj,3}));
        end
        axis([0 50 0 300]);
        quantile(squeeze(predictive(:,jj,ii)),[alpha/2,1-alpha/2])
    end
end

% add additional component for prior on probability of having a basement
prior = [0.5,0.5];betaPost = zeros(J,2);
for jj=1:J
    betaPost(jj,:) = [prior(1)+sum(radonData{jj,2}==1),prior(2)+sum(radonData{jj,2}==0)];
end

predictive = zeros(iter,J);count = 1;figure;
for jj=1:J
    countyInd = jj+1;
    firstFloor = binornd(1,betarnd(betaPost(jj,1),betaPost(jj,2)));
    for tt=1:iter
        sigsquare = posteriorSamples(tt,1);
        meanVal = posteriorSamples(tt,countyInd);
        
        if firstfloor==1
            meanVal = meanVal+posteriorSamples(tt,5);
        end
        newData = exp(normrnd(meanVal,sqrt(sigsquare)));
        
        predictive(tt,jj) = newData;
    end
    subplot(3,1,count);histogram(squeeze(predictive(:,jj)));
    title(sprintf('Predictive: %s',radonData{jj,3}));
    count = count+1;
    axis([0 50 0 350]);
    quantile(squeeze(predictive(:,jj)),[alpha/2,1-alpha/2])
end

%% PROBLEM 2: BDA3 14.11, 14.12, 14.13

data = log([[31.2;24;19.8;18.2;9.6;6.5;3.2],[10750;8805;7500;7662;5286;3724;2423],...
    [1113;982;908;842;626;430;281]]);

iter = 24e5;burnIn = 4e5;skipRate = 2000;

numParams = 5;
params = zeros(iter,numParams);

X = [ones(length(data(:,1)),1),data(:,1)];
tmp = X\data(:,3);

params(1,1) = tmp(1); % a
params(1,2) = tmp(2); % b
params(1,3) = log(var(data(:,3)-X*tmp)); % sigmsquare
params(1,4) = mean(data(:,1)); % mu
params(1,5) = log(var(data(:,1))); % tausquare

lognormpdf = @(x,mu,sigmasquare) -0.5*log(sigmasquare)-(1/(2*sigmasquare)).*(x-mu).^2;

logcauchypdf = @(x,mu,gamma) -log(pi*gamma)+2*log(gamma)-log((x-mu).^2+gamma*gamma);

abPrior = [0,10];

posterior = zeros(iter,1);

sigmasquare = exp(params(1,3));tausquare = exp(params(1,5));
posterior(1) = sum(lognormpdf(data(:,1),params(1,4),sigmasquare+tausquare))+...
    sum(lognormpdf(data(:,3),params(1,1)+params(1,2)*params(1,4),sigmasquare+params(1,2)^2*tausquare))...
   +logcauchypdf(params(1,3),0,2.5)+logcauchypdf(params(1,5),0,2.5)+lognormpdf(params(1,1),abPrior(1),abPrior(2))+...
    lognormpdf(params(1,2),abPrior(1),abPrior(2));


loglambda = log(2.38^numParams);
identity = eye(numParams);
halfSigma = cholcov(identity*loglambda);
sigma = halfSigma'*halfSigma;
updateMu = params(1,:)';
zeroVec = zeros(1,numParams);

% figure;
for ii=2:iter
    starParams = params(ii-1,:)+mvnrnd(zeroVec,sigma);
    
    sigmasquare = exp(starParams(1,3));tausquare = exp(starParams(1,5));
    starPosterior = sum(lognormpdf(data(:,1),starParams(1,4),sigmasquare+tausquare))+...
        sum(lognormpdf(data(:,3),starParams(1,1)+starParams(1,2)*starParams(1,4),sigmasquare+starParams(1,2)^2*tausquare))...
       +logcauchypdf(starParams(1,3),0,2.5)+logcauchypdf(starParams(1,5),0,2.5)...
       +lognormpdf(starParams(1,1),abPrior(1),abPrior(2))+...
        lognormpdf(starParams(1,2),abPrior(1),abPrior(2));
    
    logA = starPosterior-posterior(ii-1);
    
    if log(rand)<logA
        params(ii,:) = starParams;
        posterior(ii) = starPosterior;
    else
        params(ii,:) = params(ii-1,:);
        posterior(ii) = posterior(ii-1);
    end
    
    if mod(ii,500) == 0 && ii<burnIn
        meanSubtract = params(ii,:)'-updateMu;
        updateMu = updateMu+0.01.*meanSubtract;
        halfSigma = halfSigma+0.01.*(triu((inv(halfSigma))*(halfSigma'*halfSigma+meanSubtract*...
            meanSubtract')*((inv(halfSigma))')-identity)-halfSigma);
        sigma = halfSigma'*halfSigma;
    end
    
%     plot(ii,posterior(ii),'.');pause(1/100);hold on;
end

params = params(burnIn+1:skipRate:end,:);
params(:,3) = exp(params(:,3));
params(:,5) = exp(params(:,5));

figure;plot(posterior);

figure;
paramNames = {'a','b','\sigma^2','\mu','\tau^2'};
for ii=1:numParams
    subplot(3,2,ii);histogram(params(:,ii));
    title(sprintf('Posterior: %s',paramNames{ii}));
    quantile(params(:,ii),[alpha/2,1-alpha/2])
end

% 14.13, use both columns of data as the design
iter = 50e5;burnIn = 5e5;skipRate = 4500;

numParams = 7;
params = zeros(iter,numParams);

X = [ones(length(data(:,1)),1),data(:,1),data(:,2)];
tmp = X\data(:,3);

params(1,1) = tmp(1); % a
params(1,2) = tmp(2); % b
params(1,3) = tmp(3);
params(1,4) = log(var(data(:,3)-X*tmp)); % sigmsquare
params(1,5) = mean(data(:,1)); % mu
params(1,6) = mean(data(:,2));
params(1,7) = log(var(data(:,1))+var(data(:,2))); % tausquare

lognormpdf = @(x,mu,sigmasquare) -0.5*log(sigmasquare)-(1/(2*sigmasquare)).*(x-mu).^2;

logcauchypdf = @(x,mu,gamma) -log(pi*gamma)+2*log(gamma)-log((x-mu).^2+gamma*gamma);

abPrior = [0,10];

posterior = zeros(iter,1);

sigmasquare = exp(params(1,4));tausquare = exp(params(1,7));
posterior(1) = sum(lognormpdf(data(:,1),params(1,5),sigmasquare+tausquare))+...
    sum(lognormpdf(data(:,2),params(1,6),sigmasquare+tausquare))+...
    sum(lognormpdf(data(:,3),params(1,1)+params(1,2)*params(1,5)+params(1,3)*params(1,6),sigmasquare+(params(1,2)^2+params(1,3)^2)*tausquare))...
   +logcauchypdf(params(1,4),0,2.5)+logcauchypdf(params(1,7),0,2.5)+lognormpdf(params(1,1),abPrior(1),abPrior(2))+...
    lognormpdf(params(1,2),abPrior(1),abPrior(2))+lognormpdf(params(1,3),abPrior(1),abPrior(2));


loglambda = log(2.38^numParams);
identity = eye(numParams);
halfSigma = cholcov(identity*loglambda);
sigma = halfSigma'*halfSigma;
updateMu = params(1,:)';
zeroVec = zeros(1,numParams);

% figure;
for ii=2:iter
    starParams = params(ii-1,:)+mvnrnd(zeroVec,sigma);
    
    sigmasquare = exp(starParams(1,4));tausquare = exp(starParams(1,7));
    starPosterior = sum(lognormpdf(data(:,1),starParams(1,5),sigmasquare+tausquare))+...
        sum(lognormpdf(data(:,2),starParams(1,6),sigmasquare+tausquare))+...
        sum(lognormpdf(data(:,3),starParams(1,1)+starParams(1,2)*starParams(1,5)+starParams(1,3)*starParams(1,6),sigmasquare+(starParams(1,2)^2+starParams(1,3)^2)*tausquare))...
        +logcauchypdf(starParams(1,4),0,2.5)+logcauchypdf(starParams(1,7),0,2.5)+lognormpdf(starParams(1,1),abPrior(1),abPrior(2))+...
        lognormpdf(starParams(1,2),abPrior(1),abPrior(2))+lognormpdf(starParams(1,3),abPrior(1),abPrior(2));
    
    logA = starPosterior-posterior(ii-1);
    
    if log(rand)<logA
        params(ii,:) = starParams;
        posterior(ii) = starPosterior;
    else
        params(ii,:) = params(ii-1,:);
        posterior(ii) = posterior(ii-1);
    end
    
    if mod(ii,500) == 0 && ii<burnIn
        meanSubtract = params(ii,:)'-updateMu;
        updateMu = updateMu+0.01.*meanSubtract;
        halfSigma = halfSigma+0.01.*(triu((inv(halfSigma))*(halfSigma'*halfSigma+meanSubtract*...
            meanSubtract')*((inv(halfSigma))')-identity)-halfSigma);
        sigma = halfSigma'*halfSigma;
    end
    
%     plot(ii,posterior(ii),'.');pause(1/100);hold on;
end

params = params(burnIn+1:skipRate:end,:);
params(:,3) = exp(params(:,4));
params(:,5) = exp(params(:,7));

figure;plot(posterior);

figure;
paramNames = {'a','b','c','\sigma^2','\mu1','\mu2','\tau^2'};
for ii=1:numParams
    subplot(4,2,ii);histogram(params(:,ii));
    title(sprintf('Posterior: %s',paramNames{ii}));
    quantile(params(:,ii),[alpha/2,1-alpha/2])
end