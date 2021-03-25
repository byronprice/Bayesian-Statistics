function [logProbData,logAlpha] = ForwardHMM(P,EmissionDist,logPi,emission)
%ForwardHMM.m 
%   Implements the forward algorithm
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
%OUTPUTS:
%        logProbData - probability of the observed data, under the model
%        logAlpha - log of the probability of being in a given state at a
%          given moment
% Byron Price, 2021/01/02

[N,~] = size(emission);

logP = log(P);
K = size(P,1);

logAlpha = zeros(N,K);

for jj=1:K
    logxgivenz = LogMvnPDF(emission(1,:)',EmissionDist{jj,1},EmissionDist{jj,2});
    logAlpha(1,jj) = logxgivenz+logPi(jj);
end
prevAlpha = logAlpha(1,:);

for nn=2:N
    for jj=1:K
        logxgivenz = LogMvnPDF(emission(nn,:)',EmissionDist{jj,1},EmissionDist{jj,2});
        
        logVec = zeros(K,1);
        for kk=1:K
            logzgivenz = logP(kk,jj);
            logVec(kk) = logzgivenz+prevAlpha(kk);
        end
        logAlpha(nn,jj) = logxgivenz+LogSum(logVec,K);

    end
    prevAlpha = logAlpha(nn,:);
end
logProbData = LogSum(logAlpha(N,:),K);

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