function [Z,X,hk,Wtrue,Strue,Dtrue,ranktrue] = SimulateTDRData(N,M,T,P)
% SimulateTDRData.m
%   Simulate data to input to TDR_Gibbs, for algorithm testing

if nargin<1
    N = 50; %number of neurons
    T = 20; %number of time points
    M = 100; %maximum number of trials
    P = 4; %number of covariates
end

rmax = 3; % Maximum rank for demo

ranktrue = randi([1 rmax],[1 P]);% Select dimensionality at random

%% Simulation Parameters

% Parameters generating smooth time components
len = 2*ones(P,1);%length scale
rho = 1*ones(P,1); %Variance
[Wtrue, Strue, BB] = SimWeights(N,T,P,ranktrue,len,rho);

% %Plot the basis functions generated
% figure;
% for p=1:P
%     subplot(P,1,p)
%     plot(cat(2,Strue{p}))
%     xlabel('time')
%     title(['Bases for task variable ',num2str(p)])
%     box off
%     xlim([1 T])
%     
% end
% set(gcf,'color','w')
% subplot(P,1,P)
% title(['Bases for condition-independent component'])
% 
% figure;
% for p=1:P
%     subplot(P,1,p)
%     bar(cat(2,Wtrue{p}))
%     xlabel('Neuron index')
%     title(['Weights for task variable ',num2str(p)])
%     box off
%     xlim([1 n])
% end
% set(gcf,'color','w')
% subplot(P,1,P)
% title(['Weights for condition-independent component'])

% Set noise variance of each neuron
mstdnse = 1/.8;% Mean standard deviation of the noise
Dtrue = exprnd(mstdnse,[N 1]);

%Define all levels of regressors
var_uniq{1} = -2:2;
var_uniq{2} = -2:2;
var_uniq{3} = [-1 1];
var_uniq{4} = 1;

% Generate samples
X = SimConditions(var_uniq,M);% Task conditions
Y = SimPopData(X,BB,Dtrue, N,T,M);% Neuronal responses

% Randomly drop neurons on different trials so that trial numbers differ
% across neurons
hk = zeros(M,N);
Z = zeros(N,T,M);
pdrop = 0.5;% Probability of a neuron being dropped from any given trial
for mm = 1:M
    hk(mm,:) = binornd(1,1-pdrop,[1 N]);
    Z(:,:,mm) = spdiags(hk(mm,:)',0,N,N)*squeeze(Y(:,:,mm));
end
% Z = permute(Z,[1,3,2]);

end