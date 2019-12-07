function [W,S,b,D,rank] = TDR_Gibbs(Z,X,hk,rp,numSamples)
% TDR_Gibbs.m
%  Gibbs sampler for dimensionality reduction model on neural data, based on
%   "Model-Based Targeted Dimensionality Reduction for Neuronal Population
%   Data" Aoi & Pillow 2018
%
% model is Z_(nmt) = b_n + x_m1*(W_n1*S_t1) + ... + x_mp*(W_np*S_tp) +
%    noise
%  so, this is a linear regression for each neuron, but for each predictor,
%  x_mp, the neurons share basis functions S
% noise is Guassian-independent for each neuron
%
% we need to discover the rank of the W and S for each predictor, to do so
% we use a greedy algorithm that calculates the AIC
%
%INPUTS: Z - neural data, a firing rate tensor (matlab 3-D array) of size N-M-T, 
%         for N neurons, M for total trials, T time points per trial
%        X - a matrix of size M-P, containing trial-dependent predictors,
%          include a column of ones for condition-independent component
%          (could easily extend so that X is depdent on time as well)
%        hk - a one-hot matrix of size M-N denoting which neurons were recorded 
%           on which trials
%  (optional)
%        rp - initial rank of W and S matrices for each covariate, a
%          vector of size P-1, defaults to a vector of zeros 
%        numSamples - number of samples to generate from posterior,
%           defaults to 1000
%
%OUTPUTS: W - the "neuron factors", the neuron-dependent mixing weights, a
%          cell array of size numSamples-1, with another cell inside of P-1,
%          and a finally matrix of size N-rp in each cell, rp being the rank 
%          of the matrix for that predictor
%         S - a common set of time-varying basis factors, a cell array numSamples-1, 
%           then another cell P-1, with a matrix of size T-rp in each cell
%         b - baseline firing rates, a vector of size N-numSamples
%         D - firing rate variance, a vector of size N-numSamples
%
%Created: 2019/12/06
% Byron Price
%Updated: 2019/12/06
% By: Byron Price

[N,M,T] = size(Z);
hk = logical(hk);

newZ = cell(N,1); % convert Z to cell array to get rid of empty space
for nn=1:N
    newZ{nn} = squeeze(Z(nn,hk(:,nn),:));
end
Z = newZ;

P = size(X,2);

Mn = sum(hk,1); % total trials per neuron
MnT = Mn.*T;

if nargin<4
    rp = zeros(P,1);
    numSamples = 1000;
elseif nargin<5
    numSamples = 1000;
end

finalW = cell(numSamples,1);finalS = cell(numSamples,1);
finalB = zeros(N,numSamples);finalD = zeros(N,numSamples);

% PRIORS
Wprior = [0,1]; % mean and variance for normal

% everything else gets a flat prior
% Dprior = [0,0]; % nu and tau-square for scaled inverse chi square 
% Sprior = [0,Inf]; % mean and variance for normal
% bprior = [alpha,beta]; % gamma distribution would be one option

currentAIC = Inf;
aicDiff = -Inf;
bestRank = rp;

burnIn = 5e3;
numSkip = 10;numIter = numSkip*numSamples;

% GREEDY ALGORITHM TO CALCULATE OPTIMAL RANKS
while aicDiff<0
    loglikelihood = zeros(numSamples,1);
    
    acrossPAIC = currentAIC;check = 0;
    for changeRank=P:-1:1
        currentRank = rp;currentRank(changeRank) = currentRank(changeRank)+1;
        
        numParams = N*2;
        for pp=1:P
            numParams = numParams+currentRank(pp)*N+currentRank(pp)*T;
        end
        
        % semi-random initialization
        [W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,currentRank);
        
        % Gibbs sampler burn-in period
        Ztilde = ComputeSuffStats(W,S,b,Z,X,hk,N,Mn,P,T,currentRank);
        for iter=1:burnIn
            [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,P,T,currentRank,Wprior);
%             tmplikelihood = GetLikelihood(Ztilde,D,N,MnT);
%             plot(iter,tmplikelihood,'.');pause(1/100);hold on;
        end
        
        tmplikelihood = GetLikelihood(Ztilde,D,N,MnT);
        tmpAIC = 2*numParams-2*tmplikelihood;
        
        if tmpAIC<=acrossPAIC
            check = 1;
            bestRank = currentRank;
            count = 0;
            % Gibbs sampler for real
            for iter=1:numIter
                [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,P,T,currentRank,Wprior);
                
                if mod(iter,numSkip)==0
                    count = count+1;
                    %             Ztilde = ComputeSuffStats(W,S,b,Z,X,hk,N,Mn,P)
                    loglikelihood(count) = GetLikelihood(Ztilde,D,N,MnT);
                    
                    finalW{count} = W;
                    finalS{count} = S;
                    finalB(:,count) = b;
                    finalD(:,count) = D;
                end
            end
            acrossPAIC = 2*numParams-2*max(loglikelihood);
        end
    end
    if check == 1
        rp = bestRank;
        AIC = 2*numParams-2*max(loglikelihood);
        aicDiff = AIC-currentAIC;
        
        currentAIC = AIC;
    else
        aicDiff = Inf;
        break;
    end
end

W = finalW;
S = finalS;
b = finalB;
D = finalD;
rank = rp;
end

function [S,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,X,hk,rp)
% REALLY SUPER SSLLLOOOOOOWWWWWW
for pp=1:P
    if rp(pp)>0
        for tt=1:T
            for rr=1:rp(pp)
                muN = zeros(N,1);
                deltaN = zeros(N,1);
                for nn=1:N
                    tmp = X(hk(:,nn),pp)*W{pp}(nn,rr);
                    Ztilde{nn}(:,tt) = Ztilde{nn}(:,tt)+tmp*S{pp}(tt,rr);
                    
                    deltaN(nn) = sum(tmp(:).^2);
                    muN(nn) = sum(tmp.*Ztilde{nn}(:,tt))/deltaN(nn);
                    deltaN(nn) = deltaN(nn)/D(nn);
                end
                variance = 1/(sum(deltaN));
                mu = sum(deltaN.*muN)*variance;
                
                S{pp}(tt,rr) = SimulateNormal(mu,variance);
                
                for nn=1:N
                    Ztilde{nn}(:,tt) = Ztilde{nn}(:,tt)-...
                        X(hk(:,nn),pp)*W{pp}(nn,rr)*S{pp}(tt,rr);
                end
            end
        end
    end
end
end

function [W,Ztilde] = GibbsForW(W,S,D,Ztilde,X,hk,P,N,rp,Wprior)
% VERY SLOW
for pp=1:P
    if rp(pp)>0
        for nn=1:N
            for rr=1:rp(pp)
                tmp = X(hk(:,nn),pp)*S{pp}(:,rr)';
                Ztilde{nn} = Ztilde{nn}+tmp*W{pp}(nn,rr);
                
                delta = sum(tmp(:).^2);
                mu = sum(sum(tmp.*Ztilde{nn}))/delta;
                
                delta = delta/D(nn);
                variance = 1/(delta+Wprior(2));
                mu = (mu*delta)*variance; % assumes Wprior(1) = 0
                
                W{pp}(nn,rr) = SimulateNormal(mu,variance);
                
                Ztilde{nn} = Ztilde{nn}-tmp*W{pp}(nn,rr);
            end
        end
    end
end
end

function [b,Ztilde] = GibbsForB(b,D,Ztilde,MnT,N)

for nn=1:N
    Ztilde{nn} = Ztilde{nn}+b(nn);
    mu = mean(Ztilde{nn}(:));
    variance = D(nn)/MnT(nn);
    
    b(nn) = SimulateNormal(mu,variance);
    
    Ztilde{nn} = Ztilde{nn}-b(nn);
end
end

function [D] = GibbsForD(D,Ztilde,MnT,N)

for nn=1:N
    nutau = sum(Ztilde{nn}(:).^2);
    
    D(nn) = SimulateInvChiSquare(max(MnT(nn)-2,0.5),nutau);
end
end

function [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,P,T,rp,Wprior)

[newS,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,X,hk,rp);
[newW,Ztilde] = GibbsForW(W,newS,D,Ztilde,X,hk,P,N,rp,Wprior);
[newB,Ztilde] = GibbsForB(b,D,Ztilde,MnT,N);
[newD] = GibbsForD(D,Ztilde,MnT,N);

W = newW;
S = newS;
b = newB;
D = newD;
end

function [loglikelihood] = GetLikelihood(Ztilde,D,N,MnT)
loglikelihood = 0;

for nn=1:N
    tmp = Ztilde{nn}(:).^2;
    loglikelihood = loglikelihood-log(2*pi*D(nn))*(MnT(nn)/2)-...
        (1/(2*D(nn))).*sum(tmp);
end

end

function [Ztilde] = ComputeSuffStats(W,S,b,Z,X,hk,N,Mn,P,T,rp)

Ztilde = cell(N,1);

for nn=1:N
   Ztilde{nn} = Z{nn}-ones(Mn(nn),T).*b(nn);
   
   for pp=1:P
       if rp(pp)>0
          Ztilde{nn} = Ztilde{nn}-X(hk(:,nn),pp)*(W{pp}(nn,:)*S{pp}');
       end
   end
end

end

function [W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,rp)
W = cell(P,1);S = cell(P,1);
b = zeros(N,1);D = zeros(N,1);

for nn=1:N
   reduceData = Z{nn};
   b(nn) = mean(reduceData(:));
   D(nn) = var(reduceData(:));
end

for pp=1:P
   W{pp} = normrnd(Wprior(1),sqrt(Wprior(2)),[N,rp(pp)]);
   S{pp} = normrnd(0,1,[T,rp(pp)]);
end
end

function [normVal] = SimulateNormal(mu,variance)

normVal = normrnd(mu,sqrt(variance));
end

function [invchiVal] = SimulateInvChiSquare(nu,tau)
% tau is nu*tau-squared
invchiVal = 1./gamrnd(nu/2,2/tau);
end