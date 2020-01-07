function [W,S,b,D,rank,likelihood] = TDR_GibbsRF(Z,X,hk,stimDim,R,numSamples)
% TDR_GibbsRF.m
%  Gibbs sampler for dimensionality reduction model on neural data, based on
%   "Model-Based Targeted Dimensionality Reduction for Neuronal Population
%   Data" Aoi & Pillow 2018
%
% model is Z_(nm) = b_n + x_m1*(W_n1*S_1)' + ... + x_mp*(W_nP*S_P)' + noise
%  so, this is a linear regression for each neuron, but for each predictor,
%  x_mp, the neurons share basis functions S ... in this case, we only
%  assume one predictor (e.g. a visual stimulus, but we could add more)
% noise is Guassian-independent for each neuron and timepoint
%
% we need to discover the rank of W and S
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
%Created: 2020/01/07
% Byron Price
%Updated: 2020/01/07
% By: Byron Price

[N,M] = size(Z);
hk = logical(hk);

P = 1; % one covariate in this case
dr = stimDim(1);dc = stimDim(2); % dimensionality of stimulus

[XtX,XcTXc,XrTXr] = ComputeStimulusSuffStats(X,M,dr,dc);

Mn = sum(hk,1); % total trials per neuron

if nargin<5
    R = 1;
    numSamples = 1000;
elseif nargin<6
    numSamples = 1000;
end

% PRIORS
Wprior = [0,1]; % mean and variance for normal

[Winit,Sinit,~,~] = InitializeParams(Z,N,P,T,Wprior,R);

% everything else gets a flat prior
% Dprior = [0,0]; % nu and tau-square for scaled inverse chi square 
% Sprior = [0,Inf]; % mean and variance for normal
% bprior = [alpha,beta]; % gamma distribution would be one option

currentAIC = Inf;
aicDiff = -Inf;
bestRank = R;
Wbest = Winit;
Sbest = Sinit;

burnIn = 5e4;
numSkip = 10;numIter = numSkip*numSamples;
CA = 1; % run coordinate ascent to maximize likelihood
tolerance = 1e-4; % when to stop coordinate ascent based on likelihood difference

fprintf('Selecting model rank ...\n');
% GREEDY ALGORITHM TO CALCULATE OPTIMAL RANKS
while aicDiff<0
    
    acrossPAIC = currentAIC;check = 0;
    for changeRank=P:-1:1
        currentRank = R;currentRank(changeRank) = currentRank(changeRank)+1;
        
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
            
            [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,XtX,hk,N,MnT,Mn,P,T,currentRank,Wprior,Xp,CA);
            
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
        if iter==burnIn
            fprintf('Warning! Iteration limit reached ...\n');
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
        R = bestRank;
        fprintf('Current Rank: ');
        fprintf('%g',R');
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

fprintf('Rank Selection Complete\n');
fprintf('Final Rank: ');
fprintf('%g',R');
fprintf('\n');

fprintf('\nRunning Gibbs Sampler ...\n');
GS = 0; % run sampler, rather than coordinate ascent
finalW = cell(numSamples,1);finalS = cell(numSamples,1);
finalB = zeros(N,numSamples);finalD = zeros(N,numSamples);

[W,S,b,D] = InitializeParams(Z,N,P,T,Wprior,R,Winit,Sinit);

% Gibbs sampler burn-in period
Ztilde = ComputeSuffStats(W,S,b,Z,X,hk,N,P,R,Xp);
for iter=1:burnIn
    [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,XtX,hk,N,MnT,Mn,P,T,R,Wprior,Xp,GS);
end
count = 0;
% Gibbs sampler for real
for iter=1:numIter
    [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,XtX,hk,N,MnT,Mn,P,T,R,Wprior,Xp,GS);
    
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
rank = R;

likelihood = GetLikelihood(Ztilde,D,N,MnT);
end

function [S,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,Mn,X,XtX,hk,rp,Xp,CA)
% assumes precision is diagonal
% oneVec = ones(T,1);
% identity = diag(oneVec);
for pp=1:P
    if rp(pp)>0
        muS = zeros(rp(pp),T);
        precisionV = zeros(rp(pp),rp(pp));

        for nn=1:N
            if Xp(nn,pp)
                Ztilde{nn} = Ztilde{nn}+kron(X(hk(:,nn),pp),S{pp}*W{pp}(nn,:)');
                constant = XtX(nn,pp)/D(nn);
                muS = muS+constant*W{pp}(nn,:)'*(X(hk(:,nn),pp)\reshape(Ztilde{nn},[T,Mn(nn)])');

                precisionV = precisionV+W{pp}(nn,:)'*W{pp}(nn,:).*constant;
            end
        end
        
        M = precisionV\muS;
        
        if CA
            S{pp} = M';
        else
            S{pp} = SimulateMatrixNormal(M',precisionV,T,rp(pp));
        end
        
        for nn=1:N
            Ztilde{nn} = Ztilde{nn}-...
                kron(X(hk(:,nn),pp),S{pp}*W{pp}(nn,:)');
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
                
                mu = (precision\(tmp'*Ztilde{nn}))./D(nn);
                
                if CA
                    % coordinate ascent
                    W{pp}(nn,:) = mu';
                else
                    % gibbs, random sample
                    W{pp}(nn,:) = SimulateMVNormal(mu,precision,rp(pp))';
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
    
    if CA
        % coordinate ascent, take mode
        b(nn) = mu;
    else
        % gibbs, generate sample
        variance = D(nn)/MnT(nn);
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

function [W,S,b,D,Ztilde] = RunGibbs(W,S,b,D,Ztilde,X,XtX,hk,N,MnT,Mn,P,T,rp,Wprior,Xp,CA)
% CA is an indicator telling whether to do coordinate ascent (1) by taking
% the mode of the conditional distribution, or to do Gibbs sampling by
% generating a random sample
[newS,Ztilde] = GibbsForS(W,S,D,Ztilde,N,P,T,Mn,X,XtX,hk,rp,Xp,CA);
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

function [normVals] = SimulateMVNormal(mu,precision,N)
Ainv = chol(precision);
% A = InvUpperTri(Ainv);
                
normVals = mu+Ainv\randn([N,1]);
end

function [normVals] = SimulateMatrixNormal(M,Vinv,T,R)
% assume Uinv is identity
Binv = chol(Vinv,'lower');
% B = InvUpperTri(Binv')';

normVals = M+randn([T,R])/Binv; % would be M+A*X*B;
% Ainv = chol(Uinv);
% A = InvUpperTri(Ainv);
end

function [invchiVal] = SimulateInvChiSquare(nu,tau)
% tau is nu*tau-squared
invchiVal = 1./gamrnd(nu/2,2/tau);
end

function [XtX,XcTXc,XrTXr] = ComputeStimulusSuffStats(X,M,dr,dc)
X = X-sum(X,1)./M; % mean subtract
XtX = cov(X); % full covariance

% get estimator of Kronecker-structured covariance
[XrTXr,XcTXc] = KronStructCovEst(X,XtX,M,dr,dc);

end

function [XrTXr,XcTXc] = KronStructCovEst(X,XtX,M,dr,dc)
% basic idea is simple, compute across-row covariance, multivariate
%  z-score the data to remove across-row correlation structure and
%  variance, and then compute the acorss-column covariance, use those to
%  initialize algo to get ML estimate, read about flip-flop algorithm for a
%  better way to do this

% row covariance
rowForm = reshape(X',[dr,dc*M])';

XrTXr = cov(rowForm);

zscoreRow = (chol(XrTXr)')\(rowForm-mean(rowForm,1))';

% column covariance
columnForm = zeros(dc,dr*M);

bigIndex = 1;
for ii=1:dr
    for jj=1:M
         columnForm(:,bigIndex) = zscoreRow(ii,1+(jj-1)*dc:...
             dc+(jj-1)*dc)';
         bigIndex = bigIndex+1;
    end
end

XcTXc = cov(columnForm');

% ml estimator has ML proportional to 
%  -log(det(kron(XcTXc,XrTXr)))-trace(inv(kron(XcTXc,XrTXr))*XtX)
% [cholC,cholR] = MinimizeCovML(XtX,XcTXc,XrTXr,dc,dr);
% 
% XcTXc = cholC'*cholC;
% XrTXr = cholR'*cholR;
end

function [cholC,cholR] = MinimizeCovML(XtX,XcTXc,XrTXr,dc,dr)
cholCinv = inv(chol(XcTXc));cholRinv = inv(chol(XrTXr));

% cholCinv = eye(dc);cholRinv = eye(dr)*4000;
tolerance = 1e-6;

currentML = GetMLCovEst(XtX,dc,dr,cholCinv,cholRinv);
for iter=1:1000
    if mod(iter,2)==1
        
        colInv = cholCinv*cholCinv';
        rowCov = zeros(dr,dr);
        
        for jj=1:dc
            for ii=1:dc
                rowCov = rowCov+XtX(1+(ii-1)*dr:dr+(ii-1)*dr,1+(jj-1)*dr:dr+(jj-1)*dr)...
                    .*colInv(ii,jj);
            end
        end
        rowCov = rowCov./dc;
        cholRinv = inv(chol(rowCov));
        
    else   
        rowInv = cholRinv*cholRinv';
        colCov = zeros(dc,dc);
        
        for jj=1:dr
            for ii=1:dr
                colCov = colCov+XtX(ii:dr:dr*dc,jj:dr:dr*dc)*rowInv(ii,jj);
            end
        end
        colCov = colCov./dr;
        cholCinv = inv(chol(colCov));
        
    end
    tmp = GetMLCovEst(XtX,dc,dr,cholCinv,cholRinv);
    criterion = currentML-tmp;
    currentML = tmp;
    if criterion<tolerance
        break;
    end
end

cholC = inv(cholCinv)';cholR = inv(cholRinv)';
end

function [negativeLikelihood] = GetMLCovEst(XtX,dc,dr,cholCinv,cholRinv)

negativeLikelihood = -2*dr*sum(log(diag(cholCinv)))-2*dc*sum(log(diag(cholRinv)));

negativeLikelihood = negativeLikelihood+trace(kronmult({cholRinv*cholRinv',cholCinv*cholCinv'},XtX));
end


function y = kronmult(Amats,x,ii)
%  FROM AOI & PILLOW SCALABLE BAYESIAN INFERENCE
% Multiply matrix (.... A{3} kron A{2} kron A{1})(:,ii) by x
%
% y = kronmult(Amats,x,ii);
% 
% INPUT
%   Amats  - cell array with matrices {A1, ..., An}
%        x - matrix to multiply with kronecker matrix formed from Amats
%      ii  - binary vector indicating sparse locations of x rows (OPTIONAL)
%
% OUTPUT
%    y - matrix (with same number of cols as x)
%
% Equivalent to (for 3rd-order example)
%    y = (A3 \kron A2 \kron A1) * x
% or in matlab:
%    y = kron(A3,kron(A2,A1)))*x
%
% Exploits the identity that 
%    y = (A2 kron A1) * x 
% is the same as
%    y = vec( A1 * reshape(x,m,n) * A2' )
% but the latter is far more efficient.
%
% Computational cost: 
%    Given A1 [p x n] and A2 [q x m], and x a vector of length nm, 
%    standard implementation y = kron(A2,A1)*x costs O(nmpq)
%    whereas this algorithm costs O(nm(p+q))

ncols = size(x,2);

% Check if 'ii' indices passed in for inserting x into larger vector
if nargin > 2
    x0 = zeros(length(ii),ncols);
    x0(ii,:) = x;
    x = x0;
end
nrows = size(x,1);

% Number of matrices
nA = length(Amats);

if nA == 1
    % If only 1 matrix, standard matrix multiply
    y = Amats{1}*x;
else
    % Perform remaining matrix multiplies
    y = x; % initialize y with x
    for jj = 1:nA
        [ni,nj] = size(Amats{jj}); %
        y = Amats{jj}*reshape(y,nj,[]); % reshape & multiply
        y =  permute(reshape(y,ni,nrows/nj,[]),[2 1 3]); % send cols to 3rd dim & permute
        nrows = ni*nrows/nj; % update number of rows after matrix multiply
    end
    
    % reshape to column vector
    y = reshape(y,nrows,ncols);

end

end

function y = kronmultinv(Amats,x,ii)
% Multiply matrix (.... inv(A{3}) kron inv(A{2}) kron inv(A{1}))(:,ii) by x
%
% y = kronmultinv(Amats,x,ii);
% 
% INPUT
%   Amats  - cell array with matrices {A1, ..., An}
%        x - matrix to multiply with kronecker matrix formed from Amats
%      ii  - binary vector indicating sparse locations of x rows (OPTIONAL)
%
% OUTPUT
%    y - matrix (with same number of cols as x)
%
% Equivalent to (for 3rd-order example)
%    y = (inv(A3) \kron inv(A2) \kron inv(A1)) * x
% or in matlab:
%    y = kron(inv(A3),kron(inv(A2),inv(A1))))*x
%
% Exploits the identity that 
%    y = (inv(A2) kron inv(A1)) * x 
% is the same as
%    y = vec( inv(A1) * reshape(x,m,n) * inv(A2)' )
% but the latter is far more efficient.
%
% Computational cost: 
%    Given A1 [p x n] and A2 [q x m], and x a vector of length nm, 
%    standard implementation y = kron(A2,A1)*x costs O(nmpq)
%    whereas this algorithm costs O(nm(p+q))

ncols = size(x,2);

% Check if 'ii' indices passed in for inserting x into larger vector
if nargin > 2
    x0 = zeros(length(ii),ncols);
    x0(ii,:) = x;
    x = x0;
end
nrows = size(x,1);

% Number of matrices
nA = length(Amats);

if nA == 1
    % If only 1 matrix, standard matrix inverse times x
    y = Amats{1}\x;
else
    % Perform remaining matrix multiplies
    y = x; % initialize y with x
    for jj = 1:nA
        [ni,nj] = size(Amats{jj}); %
        y = Amats{jj}\reshape(y,nj,[]); % reshape & multiply by inv
        y =  permute(reshape(y,ni,nrows/nj,[]),[2 1 3]); % send cols to 3rd dim & permute
        nrows = ni*nrows/nj; % update number of rows after matrix multiply
    end
    
    % reshape to column vector
    y = reshape(y,nrows,ncols);

end
end