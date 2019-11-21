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

%% PROBLEM 2: BDA3 14.11, 14.12, 14.13

