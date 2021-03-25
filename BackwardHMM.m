function [logBeta] = BackwardHMM(P,EmissionDist,emission)
%BackwardHMM.m 
%   Implements the backward algorithm
%    given a Hidden Markov model with state transition probabilities
%   given by P and assuming the emission distribution given the state is a
%   normal random variable with a unique mean and variance for each state,
%   find the probability of the data P(y)
%INPUTS:
%        P - transition probability matrix [number of states by number of
%           states]
%        EmissionDist - emission cell array (mean and covariance of normal
%           distribution for each state) [states by 2]
%        emission - observed data
%        logAlpha - log of the probability of being in a given state a
%         given moment, provided by the forward algorithm
%OUTPUTS:
%        logBeta - log of the probability of being in a given state at a
%          given moment
% Byron Price, 2021/01/02

[N,~] = size(emission);

logP = log(P);
K = size(P,1);

logBeta = zeros(N,K);

logBeta(N,:) = 0;
prevBeta = logBeta(N,:);

for nn=N-1:-1:1
    for jj=1:K
        
        logVec = zeros(K,1);
        for kk=1:K
            logxgivenz = LogMvnPDF(emission(nn+1,:)',EmissionDist{kk,1},EmissionDist{kk,2});
            logzgivenz = logP(jj,kk);
            logVec(kk) = logxgivenz+logzgivenz+prevBeta(kk);
        end
        logBeta(nn,jj) = LogSum(logVec,K);

    end
    prevBeta = logBeta(nn,:);
end

end

function [logPDF] = LogMvnPDF(data,mu,sigma)
logdet = 2*sum(log(diag(chol(sigma))));
logPDF = -0.5*logdet-0.5*(data-mu)'*(sigma\(data-mu));

end

function [summation] = LogSum(vector,vectorLen)
if vectorLen==0
    summation = -Inf;
elseif vectorLen==1
    summation = vector(1);
else

    maxVal = max(vector);
    difference = vector-maxVal;
    summation = maxVal+log1p(sum(exp(difference))-1);
    
%     vector = sort(vector);
%     summation = LogSumExpTwo(vector(1),vector(2));
%     for ii=2:vectorLen-1
%         summation = LogSumExpTwo(summation,vector(ii+1));
%     end
end

end