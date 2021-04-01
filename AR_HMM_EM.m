function [logProbPath,states,A,Pi,P,EmissionDist,steadyState] = AR_HMM_EM(data,K,nLags)
% AR_HMM_EM.m
%   EM algorithm to fit the parameters of a discrete-state autoregressive
%     Hidden Markov Model
%
%  certain scenarios warrant a so-called left-to-right transition
%   probability matrix, in that case, simply set the matrix initially to be
%   an upper triangular matrix, if the probabilities start at zero, they
%   will remain at 0

[N,d] = size(data);

nBackData = zeros(N-nLags,d,nLags+1);
nBackData(:,:,end) = 1;
for tt=1:nLags
    nBackData(:,:,tt) = data(nLags+1-tt:end-tt,:);
end

data = data(nLags+1:end,:);
[N,~] = size(data);

% sufficient stats
xPhiT = zeros(N,d,d*nLags+1);
phiPhiT = zeros(N,d*nLags+1,d*nLags+1);
for nn=1:N
    tmp = squeeze(nBackData(nn,:,:));
    tmp = tmp(:);tmp = tmp(1:end-d+1);
    xPhiT(nn,:,:) = log(data(nn,:)'*tmp');
    phiPhiT(nn,:,:) = log(tmp*tmp');
end

AA = squeeze(sum(real(exp(xPhiT)),1))/squeeze(sum(real(exp(phiPhiT)),1));

nBackData = reshape(nBackData,[N,d*(nLags+1)]);
nBackData = nBackData(:,1:end-d+1);

% start with Gaussian Mixture Model to initialize emission distribution and
%  transition probability matrix
[Pi,~,sigma,~] = GaussMixtureEM(data,K);

logPi = log(Pi);

% states = zeros(N,1);
% 
% for nn=1:N
%     [~,ind] = max(logalpha(nn,:));
%     states(nn) = ind;
% end

% newsigma = zeros(d,d);
% for kk=1:K
%     newsigma = newsigma+real(sigma{kk})./K;
% end
% sigma = newsigma;

Id = eye(d);
EmissionDist = cell(K,2);A = cell(K,1);
mu = cell(K,1);
for kk=1:K
    A{kk} = AA;
    mu{kk} = nBackData*AA';
    EmissionDist{kk,1} = mu{kk};
    EmissionDist{kk,2} = sigma{kk}\Id;
end
clear AA;

pseudoObservations = 100; % for dirichlet prior
stickiness = 0.925;
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
    %   calculate responsibilities and soft transition matrix
    [currentLikelihood,logalpha,logbeta] = ForwardBackwardHMM(P,EmissionDist,logPi,data);
    
    logepsilon = zeros(N-1,K,K);
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
   
    % emissions update, state emission distribution
%     sigma = zeros(d,d);
    for kk=1:K
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
        
        A{kk} = sumXPhiT/sumPhiPhiT;
        mu{kk} = nBackData*A{kk}';
        
        tmp = zeros(N,d,d);
        for nn=1:N
            tmp(nn,:,:) = log((data(nn,:)'-mu{kk}(nn,:)')*(data(nn,:)'-mu{kk}(nn,:)')')+...
                logalpha(nn,kk)+logbeta(nn,kk);
        end
        
        for jj=1:d
            for dd=1:d
                sigma{kk}(jj,dd) = real(exp(LogSum(squeeze(tmp(:,jj,dd)),N)...
                    -normalization));
%                   sigma(jj,dd) = sigma(jj,dd)+LogSum(squeeze(tmp(:,jj,dd)),N);
            end
        end
        
        EmissionDist{kk,1} = mu{kk};
        EmissionDist{kk,2} = sigma{kk}\Id;
    end
    
%     normalization = LogSum(logalpha(:)+logbeta(:),N*K);
%     sigma = exp(sigma-normalization);
%     sigmaInv = sigma\Id;
%     for kk=1:K
%         EmissionDist{kk,2} = sigmaInv;
%     end

%     [currentLikelihood,~] = ForwardHMM(P,EmissionDist,Pi,data);
    currentLikelihood-prevLikelihood
    if currentLikelihood-prevLikelihood > tolerance
        prevLikelihood = currentLikelihood;
    else
        break;
    end
end

% calculate most probable sequence of states
[logProbPath,states] = ViterbiHMM(P,EmissionDist,logPi,data);
Pi = exp(logPi);

% calculate steady state probabilities
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

function [logProbPath,states] = ViterbiHMM(P,EmissionDist,logPi,emission)
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
[logProbData,~] = ForwardHMM(P,EmissionDist,logPi,emission);
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