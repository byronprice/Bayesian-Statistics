function [logProbPath,states,A,Pi,P,EmissionDist,steadyState] = AR_HMM_EM(data,K,nLags,alpha)
% AR_HMM_EM.m
%   EM algorithm to fit the parameters of a discrete-state autoregressive
%     Hidden Markov Model
%    - a discrete first-order Markov chain governs the latent state space, 
%      while a vector autoregressive process models the observed data
%
%INPUT:  data - raw data (N-by-d matrix where N is the number of time points 
%           and d is the dimensionality
%           (in this case, N is the number of video frames and d is 10 for 10
%           principal components extracted from pre-processed Kinect2 depth
%           video)
%        K - desired number of states / behavioral modules
%        nLags - number of lags to use for autoregressive process 
%        alpha - constant between 0 and 1 for normalization of the observed
%        data covariance ... the covariance for each latent state will
%            be (1-alpha)*Sigma_global + (alpha)*Sigma_local ... this
%            regularizes the covariance matrices across states, alpha = 1
%            means the algorithm uses only the local (within-state) covariance
%               Defaults to 1. You can find the best value through
%               cross-validation.
%
%OUTPUT: logProbPath - log probability of the most likely path through the
%           latent state space, output by the Viterbi algorithm
%        states - most likely sequence of states for each frame in the
%           input data (a number from 1 to K in each of (N-nLags) positions 
%           denoting the estimated latent space)
%        A - cell array (K-by-1) where each cell contains a set of autoregressive 
%           transition matrices for each of the K states and nLags time
%           lags, estimated from the data
%        Pi - K-by-1 vector, estimated categorical probability distribution 
%           representing probability for the first state in the sequence 
%           (to initialize Markov chain)
%        P - K-by-K matrix, learned transition probability matrix for the latent
%            Markov chain
%        EmissionDist - K-by-2 cell array, where each cell represents the
%            inferred mean and precision (inverse covariance) of the 
%            autoregressive emission distribution for each state
%        steadyState - K-by-1 vector, inferred steady state categorical probability
%            distribution representing the long-run probability that the
%            latent process is in a given state (among K states)

if nargin<4
    alpha = 1; % use local covariance
end

[N,d] = size(data);

nBackData = zeros(N-nLags,d,nLags+1); % reorganize data for autoregressive process,
                        % such that we get the data at different time lags
                        % back
nBackData(:,:,end) = 1;
for tt=1:nLags
    nBackData(:,:,tt) = data(nLags+1-tt:end-tt,:);
end

data = data(nLags+1:end,:);
[N,~] = size(data);

% compute sufficient stats for M step
xPhiT = zeros(N,d,d*nLags+1);
phiPhiT = zeros(N,d*nLags+1,d*nLags+1);
for nn=1:N
    tmp = squeeze(nBackData(nn,:,:));
    tmp = tmp(:);tmp = tmp(1:end-d+1);
    xPhiT(nn,:,:) = log(data(nn,:)'*tmp');
    phiPhiT(nn,:,:) = log(tmp*tmp');
end

AA = squeeze(sum(real(exp(xPhiT)),1))/squeeze(sum(real(exp(phiPhiT)),1));
       % initial estimate based on all the data for the autoregressive
       % transition matrices

nBackData = reshape(nBackData,[N,d*(nLags+1)]);
nBackData = nBackData(:,1:end-d+1);

% start with Gaussian Mixture Model to initialize emission distribution and
%  Pi
[Pi,~,sigma,~] = GaussMixtureEM(data,K);

logPi = log(Pi);

Id = eye(d);
EmissionDist = cell(K,2);
A = cell(K,1); % initialize stored parameters for output
mu = cell(K,1);
for kk=1:K
    A{kk} = AA;
    mu{kk} = nBackData*AA';
    EmissionDist{kk,1} = mu{kk};
    EmissionDist{kk,2} = sigma{kk}\Id;
end
clear AA;

pseudoObservations = 50; % dirichlet prior on transition probability matrix, used in 
              % original Wiltschko paper to regularize model output
              % (in a Markov chain, the maximum likelihood estimate for two 
              %  states for which there are no examples of a transition is
              %  p_jk = 0, which in log probability is -infinity ... for
              %  the latent Markov chain, a similar problem occurs when two
              %  states are very unlikely to transition into each other,
              %  the dirichlet adds "pseudo-observations" as if one
              %  state had transitioned into the other, just a small number
              %  to avoid a probability of zero)
stickiness = 0.95;%0.925;
P = stickiness.*eye(K);
P = P+((1-stickiness)/(K-1)).*(1-eye(K));

for kk=1:K
    P(kk,:) = P(kk,:)./sum(P(kk,:));
end

maxIter = 1e4;
tolerance = 1e-6;
prevLikelihood = -Inf;
for iter=1:maxIter
    % E step
    %   Compute expectations
    %     calculate responsibilities (in log space) and soft transition matrix
    %     using the Forward-Backward algorithm
    [currentLikelihood,logalpha,logbeta] = ForwardBackwardHMM(P,EmissionDist,logPi,data);
    
    logepsilon = zeros(N-1,K,K); % soft transition matrix
    for nn=2:N
        for kk=1:K
            logxgivenz = LogMvnPDF(data(nn,:)',...
                EmissionDist{kk,1}(nn,:)',EmissionDist{kk,2});
            for jj=1:K
                logzgivenz = log(P(jj,kk));
                logepsilon(nn-1,jj,kk) = logalpha(nn-1,jj)+logxgivenz+...
                    logzgivenz+logbeta(nn,kk);
            end
        end
    end
    
    % M step
    %    update model parameters
    
    % pi update, probability of first state to start the chain
    normalization = LogSum(logalpha(1,:)+logbeta(1,:),K);
    for kk=1:K
        logPi(kk) = logalpha(1,kk)+logbeta(1,kk)-normalization;
    end
    
    % P update, transition probability matrix, add dirichlet prior to avoid probs of
    %    zero
    for jj=1:K
        dirichlet = ones(K,1)*((1-stickiness)/(K-1));dirichlet(jj) = stickiness;
        dirichlet = dirichlet*pseudoObservations;
        tmp = squeeze(logepsilon(:,jj,:));
%         normalization = LogSum(tmp(:),(N-1)*K);
        for kk=1:K
            P(jj,kk) = exp(LogSum(tmp(:,kk),N-1)-currentLikelihood)+dirichlet(kk);
        end
        P(jj,:) = P(jj,:)./sum(P(jj,:));
    end
   
    % emissions update, state emission distribution and autoregression
    %  transition matrices
    globalSigma = zeros(d,d);
    for kk=1:K
        
        % weight sufficient statistics with log responsibilities to compute
        %  autoregressive transition matrix A
        normalization = LogSum(logalpha(:,kk)+logbeta(:,kk),N);
        sumXPhiT = zeros(d,d*nLags+1);
        
        for dd=1:d
            for gg=1:d*nLags+1
                sumXPhiT(dd,gg) = real(exp(LogSum(logalpha(:,kk)+logbeta(:,kk)+squeeze(xPhiT(:,dd,gg)),N)...
                    -normalization));
            end
        end
        
        sumPhiPhiT = zeros(d*nLags+1,d*nLags+1);
        for gg=1:d*nLags+1
            for hh=1:d*nLags+1
                sumPhiPhiT(gg,hh) = real(exp(LogSum(logalpha(:,kk)+logbeta(:,kk)+squeeze(phiPhiT(:,gg,hh)),N)...
                    -normalization));
            end
        end
        
        A{kk} = sumXPhiT/sumPhiPhiT; % calculate A for each state
        mu{kk} = nBackData*A{kk}'; % mean of Gaussian emission distribution, given 
                             % value of A
        
        % compute sigma, covariance matrix of Gaussian emission
        % distribution for each state k ... computed using a weighted sum
        % of model residuals, where everything is transformed to log space
        tmp = zeros(N,d,d);
        for nn=1:N
            tmp(nn,:,:) = log((data(nn,:)'-mu{kk}(nn,:)')*(data(nn,:)'-mu{kk}(nn,:)')')+...
                logalpha(nn,kk)+logbeta(nn,kk);
        end
        
        for jj=1:d
            for dd=1:d
                tmp = LogSum(squeeze(tmp(:,jj,dd)),N);
                sigma{kk}(jj,dd) = real(exp(tmp-normalization));
                globalSigma(jj,dd) = sigma(jj,dd)+tmp;
            end
        end
        
        EmissionDist{kk,1} = mu{kk};
        EmissionDist{kk,2} = sigma{kk};
    end
    
    normalization = LogSum(logalpha(:)+logbeta(:),N*K);
    globalSigma = exp(globalSigma-normalization);
    for kk=1:K
        EmissionDist{kk,2} = (alpha*EmissionDist{kk,2}+(1-alpha)*globalSigma)\Id;
    end
    
    badInds = false(K,1);
    for kk=1:K
       try
           chol(EmissionDist{kk,2});
       catch
           prevLikelihood = -Inf;
           K = K-1;
           badInds(kk) = true;
       end
    end
    
    logPi(badInds) = [];
    P(badInds,:) = [];
    P(:,badInds) = [];
    EmissionDist(badInds,:) = [];
    A(badInds) = [];
    mu(badInds) = [];
    
    % stop criterion, change in loglikelihood less than tolerance

    currentLikelihood-prevLikelihood
    if currentLikelihood-prevLikelihood > tolerance
        prevLikelihood = currentLikelihood;
    else
        break;
    end
end

% calculate most probable sequence of states using Viterbi algorithm
logProbData = ForwardBackwardHMM(P,EmissionDist,logPi,data);
[logProbPath,states] = ViterbiHMM(P,EmissionDist,logPi,data,logProbData);
Pi = exp(logPi);

% calculate steady state probabilities from inferred transition probability
%  matrix
[~,D,V] = eig(P);
D = diag(D);
[~,index] = min(abs(D-1));
steadyState = V(:,index)./sum(V(:,index));

end

function [logProbData,logAlpha,logBeta] = ForwardBackwardHMM(P,EmissionDist,logPi,emission)
%ForwardBackwardHMM.m 
%   Implements the forward-backward algorithm
%    given a Hidden Markov model with state transition probabilities
%   given by P and assuming the emission distribution given the state is a
%   normal random variable with a unique mean and variance for each state,
%   find the probability of the data P(x)
%INPUTS:
%        P - transition probability matrix [number of states by number of
%           states]
%        EmissionDist - emission cell array (mean and covariance of normal
%           distribution for each state) [states by 2]
%        emission - observed data
%OUTPUTS:
%        logProbData - probability of the observed data, under the model
%        logAlpha - log of the probability of being in a given state at a
%          given moment from forward pass
%        logBeta - log probability of being in a given state from backward
%          pass
% Byron Price, 2021/04/01

% see attached Methods for details on algorithm

[N,~] = size(emission);

logP = log(P);
K = size(P,1);

logAlpha = zeros(N,K);
logBeta = zeros(N,K);
logxgivenz = zeros(N,K);

for jj=1:K
    logxgivenz(1,jj) = LogMvnPDF(emission(1,:)',EmissionDist{jj,1}(1,:)',EmissionDist{jj,2});
    logAlpha(1,jj) = logxgivenz(1,jj)+logPi(jj);
end

% logBeta(N,:) = 0;
backOrder = N:-1:1;

for nn=2:N
    if nn<=ceil(N/2)+1
        for jj=1:K
            logxgivenz(nn,jj) = LogMvnPDF(emission(nn,:)',EmissionDist{jj,1}(nn,:)',EmissionDist{jj,2});
            logxgivenz(backOrder(nn)+1,jj) = LogMvnPDF(emission(backOrder(nn)+1,:)',...
                EmissionDist{jj,1}(backOrder(nn)+1,:)',EmissionDist{jj,2});
        end
    end
    
    for jj=1:K
        
        logVec = zeros(K,2);
        for kk=1:K
            logzgivenz = logP(kk,jj);
            logVec(kk,1) = logzgivenz+logAlpha(nn-1,kk);

            logzgivenz = logP(jj,kk);
            logVec(kk,2) = logxgivenz(backOrder(nn)+1,kk)+logzgivenz+logBeta(backOrder(nn)+1,kk);
        end
        logAlpha(nn,jj) = logxgivenz(nn,jj)+LogSum(logVec(:,1),K);
        
        logBeta(backOrder(nn),jj) = LogSum(logVec(:,2),K);
        
    end
end
logProbData = LogSum(logAlpha(N,:),K);

end

function [logProbPath,states] = ViterbiHMM(P,EmissionDist,logPi,emission,logProbData)
%ViterbiHMM.m   
%   Implements the Viterbi algorithm
%    given a Hidden Markov model with state transition probabilities
%   given by P and assuming the emission distribution given the state is a
%   normal random variable with a unique mean and variance for each state,
%   find the most likely sequence of hidden states
%INPUTS:
%        P - transition probability matrix [number of states by number of
%           states]
%        Emissions - emission cell array (mean and variance of normal
%           distribution for each state), [number of states by 2]
%        emission - observed data (N by d)
%OUTPUTS:
%        logProbPath - probability of maximum-likelihood path, given the
%           observed data
%        states - the most likely sequence of hidden states
% Byron Price, 2020/01/02
N = size(emission,1);

logP = log(P);
K = size(P,1);

% logNormPDF = @(x,mu,sigmasquare) -0.5*log(2*pi*sigmasquare)-((x-mu).^2)./(2*sigmasquare);

V = zeros(N,K);
B = zeros(N,K);

for kk=1:K
    logxgivenz = LogMvnPDF(emission(1,:)',EmissionDist{kk,1}(1,:)',EmissionDist{kk,2});
    logzgivenz = logPi(kk);
    
    V(1,kk) = logxgivenz+logzgivenz;
    B(1,kk) = 0;
end

for ii=2:N
    for jj=1:K
        logxgivenz = LogMvnPDF(emission(ii,:)',EmissionDist{jj,1}(ii,:)',EmissionDist{jj,2});
        
        logVec = zeros(K,1);

        for kk=1:K
            logzgivenz = logP(kk,jj);
            logVec(kk) = logzgivenz+logxgivenz+V(ii-1,kk);
        end
        [val,ind] = max(logVec);
        V(ii,jj) = val;
%         [~,ind] = max(logVec2);
        B(ii,jj) = ind;
    end
end
[val,ind] = max(V(end,:));
% [logProbData,~] = ForwardHMM(P,EmissionDist,logPi,emission);
logProbPath = val-logProbData;

% backtrace
states = zeros(N,1);

states(N) = ind;
for ii=N-1:-1:1
    states(ii) = B(ii+1,states(ii+1));
end


end

function [logPDF] = LogMvnPDF(data,mu,sigmaInv)
logdet = sum(log(diag(chol(sigmaInv))));
logPDF = logdet-0.5*(data-mu)'*(sigmaInv*(data-mu));

end

function [summation] = LogSum(vector,vectorLen)
if vectorLen==0
    summation = -Inf;
elseif vectorLen==1
    summation = vector(1);
else
%     vector = real(vector);
    maxVal = max(vector);
    difference = vector-maxVal;
    summation = maxVal+log1p(sum(exp(difference))-1);
    
    if isnan(summation) || summation>1e9
        vector = sort(vector);
        summation = LogSumExpTwo(vector(1),vector(2));
        for ii=2:vectorLen-1
            summation = LogSumExpTwo(summation,vector(ii+1));
        end
    end
end

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
% if x<-34 % condition for small x
%    y = 0;
% else
%    y = log(1+exp(-x))+x; % numerically stable calculation of log(1+exp(x))
% end

y = log(1+exp(-x))+x;
y(x<=-34) = 0;

end