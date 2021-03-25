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
    logxgivenz = LogNormPDF(emission(1,:)',EmissionDist{kk,1},EmissionDist{kk,2});
    logzgivenz = logPi(kk);
    
    V(1,kk) = logxgivenz+logzgivenz;
    B(1,kk) = 0;
end

for ii=2:N
    for jj=1:K
        logxgivenz = LogNormPDF(emission(ii,:)',EmissionDist{jj,1},EmissionDist{jj,2});
        
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

function [logPDF] = LogNormPDF(data,mu,sigma)
logdet = sum(log(diag(chol(sigma))));
logPDF = -logdet-0.5*(data-mu)'*(sigma\(data-mu));

end