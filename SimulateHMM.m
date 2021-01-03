function [state,emission] = SimulateHMM(P,EmissionDist,N,start)
%SimulateHMM.m   Project 5, 1
%   Simulate a Hidden Markov model with state transition probabilities
%   given by P and assuming the emission distribution given the state is a
%   normal random variable with a unique mean and variance for each state.
%INPUTS:
%        P - transition probability matrix [number of states by number of
%           states]
%        EmissionDist - emission cell array (mean and variance of normal
%           distribution for each state), [number of states by 2]
%        N - number of transitions to simulate (length of the chain)
%        start - state to start in, defaults to 2
%OUTPUTS:
%        state - vector [N-by-1] of hidden state labels
%        emission - vector [N-by-d] of observed data, given the hidden
%        states
% Byron Price, 2018/11/25

d = length(EmissionDist{1,1});

if nargin<4
    start = 2;
end
state = zeros(N,1);
emission = zeros(N,d);

K = size(P,1);

% guarantee P is normalized properly
for kk=1:K
   P(kk,:) = P(kk,:)./sum(P(kk,:)); 
end

prevState = zeros(1,K);prevState(start) = 1;
for ii=1:N
    transitionProbs = prevState*P;
    prevState = mnrnd(1,transitionProbs);
    
    [~,ind] = find(prevState);
    state(ii) = ind;
    
    emission(ii,:) = mvnrnd(EmissionDist{ind,1},EmissionDist{ind,2});
end
end

