function [W,S,b,D,rank,likelihood] = TDR_Gibbs(Z,X,hk,rp,numSamples)
% TDR_Gibbs.m
%  Gibbs sampler for dimensionality reduction model on neural data, based on
%   "Model-Based Targeted Dimensionality Reduction for Neuronal Population
%   Data" Aoi & Pillow 2018
%
% model is Z_(nmt) = b_n + x_m1*(W_n1*S_t1) + ... + x_mp*(W_nP*S_tP) + noise
%  so, this is a linear regression for each neuron, but for each predictor,
%  x_mp, the neurons share basis functions S
% noise is Guassian-independent for each neuron and timepoint
%
% we need to discover the rank of the W and S for each predictor, to do so
% we use a greedy algorithm that calculates the AIC
%
%INPUTS: Z - neural data, a firing rate tensor (matlab 3-D array) of size N-T-M, 
%         for N neurons, M for total trials, T time points per trial
%        X - a matrix of size M-P, containing trial-dependent predictors,
%          include a column of ones for condition-independent component
%          (could easily extend so that X is depdent on time as well)
%        hk - a one-hot matrix of size M-N denoting which neurons were recorded 
%           on which trials
%  (optional)
%        rp - initial rank of W and S matrices for each covariate, a
%          vector of size P-1, defaults to a vector of zeros with 1 in
%          final position (for condition-independent covariate)
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
%         rank - P-1, rank of the factorization for each covariate
%         likelihood - the full log likelihood of the model evaluated at one of the
%           posterior samples
%
%Created: 2019/12/06
% Byron Price
%Updated: 2019/12/08
% By: Byron Price

[N,T,M] = size(Z);
hk = logical(hk);

newZ = cell(N,1); % convert Z to cell array to get rid of empty space
for nn=1:N
    tmp = squeeze(Z(nn,:,hk(:,nn)));
    newZ{nn} = tmp(:); % vec operator
end
Z = newZ;

P = size(X,2);

% create matrix to control for case that neuron has no exemplars of a
%   particular covariate
Xp = zeros(N,P);

tolerance = 1e-6;
for nn=1:N
   for pp=1:P
       if sum(abs(X(hk(:,nn),pp)))>tolerance
           Xp(nn,pp) = 1;
       end
   end
end

Np = sum(Xp,1);

Xp = logical(Xp);

Mn = sum(hk,1); % total trials per neuron
MnT = Mn.*T;

if nargin<4
    rp = zeros(P,1);rp(P) = 1;
    numSamples = 1000;
elseif nargin<5
    numSamples = 1000;
end

% PRIORS
Wprior = [0,1]; % mean and variance for normal

[Winit,Sinit,~,~] = InitializeParams(Z,N,P,T,Wprior,rp);

% everything else gets a flat prior
% Dprior = [0,0]; % nu and tau-square for scaled inverse chi square 
% Sprior = [0,Inf]; % mean and variance for normal
% bprior = [alpha,beta]; % gamma distribution would be one option

currentAIC = Inf;
aicDiff = -Inf;
bestRank = rp;
Wbest = Winit;
Sbest = Sinit;

burnIn = 5e4;
numSkip = 10;numIter = numSkip*numSamples;
CA = 1; % run coordinate ascent to maximize likelihood
tolerance = 1e-4; % when to stop coordinate ascent based on likelihood difference

% GREEDY ALGORITHM TO CALCULATE OPTIMAL RANKS
while aicDiff<0
    
    acrossPAIC = currentAIC;check = 0;
    for changeRank=P:-1:1
        currentRank = rp;currentRank(changeRank) = currentRank(changeRank)+1;
        
        numParams = N*2;
        for pp=1:P
            numParams = numParams+currentRank(pp)*(Np(pp)+T);
        end
        
        % semi-random initialization
        [W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,currentRank,Winit,Sinit);
        
        % Gibbs sampler burn-in period, to get AIC
        Ztilde = ComputeSuffStats(W,S,b,Z,X,hk,N,P,currentRank,Xp);
        likelihood = GetLikelihood(Ztilde,D,N,MnT);likelihoodDiff = Inf;
        for iter=1:burnIn
            [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,Mn,P,T,currentRank,Wprior,Xp,CA);
            
            if mod(iter,numSkip)==0
                tmplikelihood = GetLikelihood(Ztilde,D,N,MnT);
                
                likelihoodDiff = tmplikelihood-likelihood;
                likelihood = tmplikelihood;
%                 plot(iter,tmplikelihood,'.');pause(1/100);hold on;
            end
            
            if likelihoodDiff<tolerance
                break;
            end
        end
        
        tmpAIC = 2*numParams-2*likelihood;
        
        if tmpAIC<=acrossPAIC
            check = 1;
            bestRank = currentRank;
            
            acrossPAIC = tmpAIC;
            
            Wbest = W;
            Sbest = S;
        end
    end
    if check == 1
        rp = bestRank;
        fprintf('Current Rank: ');
        fprintf('%g',rp');
        fprintf('\n');
        AIC = acrossPAIC;
        aicDiff = AIC-currentAIC;
        
        currentAIC = AIC;
        
        Winit = Wbest;
        Sinit = Sbest;
    else
        aicDiff = Inf;
        break;
    end
end

GS = 0; % run sampler, rather than coordinate ascent
finalW = cell(numSamples,1);finalS = cell(numSamples,1);
finalB = zeros(N,numSamples);finalD = zeros(N,numSamples);

[W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,rp,Winit,Sinit);

% Gibbs sampler burn-in period
Ztilde = ComputeSuffStats(W,S,b,Z,X,hk,N,P,rp,Xp);
for iter=1:burnIn
    [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,Mn,P,T,rp,Wprior,Xp,GS);
end
count = 0;
% Gibbs sampler for real
for iter=1:numIter
    [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,Mn,P,T,rp,Wprior,Xp,GS);
    
    if mod(iter,numSkip)==0
        count = count+1;
        
        finalW{count} = W;
        finalS{count} = S;
        finalB(:,count) = b;
        finalD(:,count) = D;
    end
end

W = finalW;
S = finalS;
b = finalB;
D = finalD;
rank = rp;

likelihood = GetLikelihood(Ztilde,D,N,MnT);
end

function [S,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,Mn,X,hk,rp,Xp,CA)
% assumes precision is diagonal
% identity = eye(T);
oneVec = ones(T,1);
for pp=1:P
    if rp(pp)>0
        for rr=1:rp(pp)
            muS = zeros(T,1);
            precisionS = zeros(T,1);
            for nn=1:N
                if Xp(nn,pp)
                    Ztilde{nn} = Ztilde{nn}+kron(X(hk(:,nn),pp),W{pp}(nn,rr)*S{pp}(:,rr));
                    
                    % as written in the main text, but it's slow due to many
                    %  multiplications by zero
                    %                 muS = muS+kron(X(hk(:,nn),pp),W{pp}(nn,rr)*identity)'*Ztilde{nn}./D(nn);
                    muS = muS+reshape(Ztilde{nn},[T,Mn(nn)])*X(hk(:,nn),pp)*(W{pp}(nn,rr)/D(nn));
                    
                    precisionS = precisionS+...
                        (((X(hk(:,nn),pp)*W{pp}(nn,rr))'*(X(hk(:,nn),pp)*W{pp}(nn,rr))).*oneVec)./D(nn);
                    %  multiply by identity to get precision matrix, rather
                    %  than precision diagonal
                end
            end
            % for the general case where the precision is not diagonal
%             R = chol(precisionS);
%             Rinv = InvUpperTri(R);
%             
%             sigma = Rinv*Rinv';
% 
%             sigma = diag(1./diag(precisionS)); 
%             mu = sigma*muS;
            
%             S{pp}(:,rr) = SimulateMVNormal(mu,sigma)';

            sigma = 1./precisionS; 
            mu = diag(sigma)*muS;
            
            if CA
                S{pp}(:,rr) = mu;
            else
                S{pp}(:,rr) = SimulateNormal(mu,sigma);
            end
            
            for nn=1:N
                Ztilde{nn} = Ztilde{nn}-...
                    kron(X(hk(:,nn),pp),W{pp}(nn,rr)*S{pp}(:,rr));
            end
        end
    end
end
end

function [W,Ztilde] = GibbsForW(W,S,D,Ztilde,X,hk,P,N,rp,Wprior,Xp,CA)

for pp=1:P
    if rp(pp)>0
        identity = eye(rp(pp));
        for nn=1:N
            if Xp(nn,pp)
                tmp = kron(X(hk(:,nn),pp),S{pp});
                Ztilde{nn} = Ztilde{nn}+tmp*W{pp}(nn,:)';
                
                precision = (tmp'*tmp)./D(nn)+identity.*Wprior(2);
                R = chol(precision);
                Rinv = InvUpperTri(R);
                
                sigma = Rinv*Rinv';
                mu = (sigma*tmp'*Ztilde{nn})./D(nn);
                
                if CA
                    % coordinate ascent
                    W{pp}(nn,:) = mu';
                else
                    % gibbs, random sample
                    W{pp}(nn,:) = SimulateMVNormal(mu,sigma);
                end
                
                Ztilde{nn} = Ztilde{nn}-tmp*W{pp}(nn,:)';
            end
        end
    end
end
end

function [b,Ztilde] = GibbsForB(b,D,Ztilde,MnT,N,CA)

for nn=1:N
    Ztilde{nn} = Ztilde{nn}+b(nn);
    mu = mean(Ztilde{nn});
    variance = D(nn)/MnT(nn);
    
    if CA
        % coordinate ascent, take mode
        b(nn) = mu;
    else
        % gibbs, generate sample
        b(nn) = SimulateNormal(mu,variance);
    end
    
    
    Ztilde{nn} = Ztilde{nn}-b(nn);
end
end

function [D] = GibbsForD(D,Ztilde,MnT,N,CA)

% coordinate ascent, take mode as value
if CA
    for nn=1:N
        D(nn) = Ztilde{nn}'*Ztilde{nn}/MnT(nn);
    end
else
% gibbs
    for nn=1:N
        D(nn) = SimulateInvChiSquare(max(MnT(nn)-2,1),Ztilde{nn}'*Ztilde{nn});
    end
end
end

function [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,hk,N,MnT,Mn,P,T,rp,Wprior,Xp,CA)
% CA is an indicator telling whether to do coordinate ascent (1) by taking
% the mode of the conditional distribution, or to do Gibbs sampling by
% generating a random sample
[newS,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,Mn,X,hk,rp,Xp,CA);
[newW,Ztilde] = GibbsForW(W,newS,D,Ztilde,X,hk,P,N,rp,Wprior,Xp,CA);
[newB,Ztilde] = GibbsForB(b,D,Ztilde,MnT,N,CA);
[newD] = GibbsForD(D,Ztilde,MnT,N,CA);

W = newW;
S = newS;
b = newB;
D = newD;
end

function [loglikelihood] = GetLikelihood(Ztilde,D,N,MnT)
loglikelihood = 0;
twopi = 2*pi;
for nn=1:N
    loglikelihood = loglikelihood-log(twopi*D(nn))*(MnT(nn)/2)-...
        (1/(2*D(nn)))*(Ztilde{nn}'*Ztilde{nn});
end

end

function [Ztilde] = ComputeSuffStats(W,S,b,Z,X,hk,N,P,rp,Xp)
% we calculate the residual at the beginning using the random parameter
%  settings, then before we update the parameters we add the
%  current values back in (eliminating them from the residual) and then
%  subtract the new values to get the full residual again
Ztilde = cell(N,1);

for nn=1:N
   Ztilde{nn} = Z{nn}-b(nn);
   
   for pp=1:P
       if rp(pp)>0 && Xp(nn,pp)
          Ztilde{nn} = Ztilde{nn}-kron(X(hk(:,nn),pp),S{pp}*W{pp}(nn,:)');
       end
   end
end

end

function [W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,rp,Winit,Sinit)
W = cell(P,1);S = cell(P,1);
b = zeros(N,1);D = zeros(N,1);

for nn=1:N
   reduceData = Z{nn};
   b(nn) = mean(reduceData(:));
   D(nn) = var(reduceData(:));
end

if nargin<=6
    for pp=1:P
        W{pp} = normrnd(Wprior(1),sqrt(Wprior(2)),[N,rp(pp)]);
        S{pp} = normrnd(0,1,[T,rp(pp)]);
    end
else
    for pp=1:P
        oldRank = size(Winit{pp},2);
        if rp(pp)==oldRank
            W{pp} = Winit{pp};
            S{pp} = Sinit{pp};
        elseif rp(pp)>oldRank
            W{pp} = [Winit{pp},normrnd(Wprior(1),sqrt(Wprior(2)),[N,1])];
            S{pp} = [Sinit{pp},normrnd(0,1,[T,1])];
        end
    end
end
end

function [normVal] = SimulateNormal(mu,variance)

normVal = normrnd(mu,sqrt(variance));
end

function [normVals] = SimulateMVNormal(mu,sigma)

normVals = mvnrnd(mu,sigma);
end

function [invchiVal] = SimulateInvChiSquare(nu,tau)
% tau is nu*tau-squared
invchiVal = 1./gamrnd(nu/2,2/tau);
end

% R = chol(sigma); % R is upper triangular
% S = InvUpperTri(R);
% Sigma-Inv = S*S';
function [Inverse] = InvUpperTri(Matrix,Transpose)
%InvUpperTri.m  Project 1, 1-a
%   Code to take inverse of an upper triangular matrix
%Inputs: 
%        Matrix - an n-by-n upper triangular matrix
%     OPTIONAL
%        Tranpose - logical true or false, depending on whether or not you want to
%            return the inverse (false) or the inverse tranpose (true) ...
%            defaults to false
%
%Outputs: 
%        Inverse - the inverse (or inverse tranpose) of Matrix
%
% Created by: Byron Price
% 2018/09/21

if istriu(Matrix)==false
    fprintf('Matrix must be upper triangular\n');
    Inverse = NaN;
    return;
end

if nargin==1
    Transpose = false;
end

[n,~] = size(Matrix);

Inverse = [Matrix,eye(n)]; % create extended matrix with identity to the right

% loop through each row and run adjust operator
for ii=n:-1:1
    Inverse = Adjust(Inverse,ii);
end

Inverse = Inverse(:,n+1:end); % left side of extended matrix is now identity, 
                         % right side is the inverse

if Transpose==true
    Inverse = Inverse';
end

end

function [A] = Adjust(A,k)
% Adjust operator, uses the k-th row of A as way to eliminate values in the
% other rows of its left-hand side

A(k,:) = A(k,:)./A(k,k);

% for ii=1:k-1
%    A(ii,:) = A(ii,:)-A(ii,k).*A(k,:);
% end

A(1:k-1,:) = A(1:k-1,:)-A(1:k-1,k)*A(k,:); % matrix operation to perform
                              % same function as for loop above

end