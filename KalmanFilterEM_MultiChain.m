function [A,C,Gamma,Sigma,z,mu0,V0] = KalmanFilterEM_MultiChain(data,K,heldOutData)
%KalmanFilterEM.m
%   Fit latent variable (state-space) model for Kalman filter parameters
%    using the EM algorithm. Data is assumed to be driven by an underlying
%    vector autoregressive process
%     e.g. the hidden state space, x, has the following distribution
%          z-(t+1) = A*z-(t)+epsilon
%          where epsilon ~ N(0,Gamma)
%          
%          this process drives the observed data process,y, by
%          x-(t) = C*z-(t)+nu
%          where nu ~ N(0,Sigma)
%INPUT: data - observed data, input as a cell array, 1-by-P, where P is the
%        number of chains observed, which entry in the cell array a matrix
%        d-by-N, where N is the
%        number of observations in the chain and d is the dimensionality 
%        of each observation (observations number N can vary for each
%        chain)
%  OPTIONAL
%       heldOutData - observed data, cell array, 1-by-T, each d-by-M, which is 
%         held-out for cross-validation purposes, to avoid overfitting
%OUTPUTS:
%       A - auto-regressive(1) process transformation matrix, for the state
%        space z
%       C - transformation from hidden state space to observations
%       Gamma - covariance of the state space stochastic process
%       Sigma - covariance of the observation space process
%       z - ML estimates of the hidden states, d-by-N
%       mu0 - estimate for state space initial conditions
%       V0 - estimate for state-space covariance initial condition
%
%Created: 2018/12/11
% By: Byron Price

 % allow for multiple chains
numChains = size(data,2);

% check size of data
chainLen = zeros(numChains,1);

for ii=1:numChains
    [d,N] = size(data{ii});
    chainLen(ii) = N;
end

if nargin==1
    heldOut = false;
    K = d;
elseif nargin==2
    heldOut = false;
else
    heldOut = true;
    numHeldOut = size(heldOutData,2);
    heldOutLen = zeros(numHeldOut,1);
    for jj=1:numHeldOut
        [~,heldOutLen(jj)] = size(heldOutData{jj});
    end
end

% initialize parameters
Sigma = cov(data{1}')./2;

% estimate of observation noise covariance
if K==d
    Gamma = Sigma./2;
else
    Gamma = normrnd(0,1,[K,K]);
    Gamma = Gamma'*Gamma;
    Gamma = diag(diag(Gamma));
end

% transformation from z to x
C = normrnd(0,1,[d,K]);

% generate transformation matrix for vector autoregressive process
if K==d
    tmp1 = zeros(d,d);
    tmp2 = zeros(d,d);
    for ii=2:N
        tmp1 = tmp1+data(:,ii)*data(:,ii-1)';
        tmp2 = tmp2+data(:,ii-1)*data(:,ii-1)';
    end
    A = tmp1/tmp2;

else
    A = normrnd(0,1,[K,K]);
    A = A./norm(A).^2;
end

if K==d
    mu0 = mean(data{1},2);
    V0 = Sigma;
else
    mu0 = zeros(K,1);
    V0 = Gamma;
end

% suffStat = zeros(d,d);
% for ii=1:N
%     suffStat = suffStat+(1/N).*data(:,ii)*data(:,ii)';
% end

maxIter = 1e3;
tolerance = 1e-3;
prevLikelihood = -Inf;

% initialize EM values
mu_n = cell(numChains,1);
V_n = cell(numChains,1);
c_n = cell(numChains,1);
P_n = cell(numChains,1);
muhat_n = cell(numChains,1);
Vhat_n = cell(numChains,1);
J_n = cell(numChains,1);

for tt=1:maxIter
    % E step, get expected hidden state estimates
    for ii=1:numChains
        [mu_n{ii},V_n{ii},c_n{ii},P_n{ii}] = KalmanForwardAlgo(data{ii},A,C,Gamma,Sigma,mu0,V0,chainLen(ii),K);
    end

    if heldOut==true
        currentLikelihood = 0;
        for ii=1:numHeldOut
            [~,~,c_n_heldout,~] = KalmanForwardAlgo(heldOutData{ii},A,C,Gamma,Sigma,mu0,V0,heldOutLen(ii),K);
            currentLikelihood = currentLikelihood+sum(c_n_heldout);
        end
    else
        currentLikelihood = 0;
        for ii=1:numChains
            currentLikelihood = currentLikelihood+sum(c_n{ii});
        end
    end
    
    for ii=1:numChains
        [muhat_n{ii},Vhat_n{ii},J_n{ii}] = KalmanBackwardAlgo(A,mu_n{ii},V_n{ii},P_n{ii},chainLen(ii));
    end
    
    % calculate expectations
    Ez = cell(numChains,1);
    Ezn_zn = cell(numChains,1);
    Ezn_zn1 = cell(numChains,1);
%     Ezn1_zn = cell(numChains,1);
    for jj=1:numChains
        Ez{jj} = zeros(K,chainLen(jj));
        Ezn_zn{jj} = cell(chainLen(jj),1);
        Ezn_zn1{jj} = cell(chainLen(jj),1);
%         Ezn1_zn{jj} = cell(chainLen(jj),1);
        for ii=1:chainLen(jj)
            Ez{jj}(:,ii) = muhat_n{jj}{ii};
            if ii>1
                Ezn_zn1{jj}{ii} = J_n{jj}{ii-1}*Vhat_n{jj}{ii}+muhat_n{jj}{ii}*muhat_n{jj}{ii-1}';
%                 Ezn1_zn{jj}{ii} = J_n{jj}{ii}*Vhat_n{jj}{ii-1}+muhat_n{jj}{ii-1}*muhat_n{jj}{ii}';
            end
            Ezn_zn{jj}{ii} = Vhat_n{jj}{ii}+muhat_n{jj}{ii}*muhat_n{jj}{ii}';
        end
    end
    % M step, maximize expected log-likelihood
    
    % update initial conditions
    mu0 = Ez{1}(:,1);
    for jj=2:numChains
        mu0 = mu0+Ez{jj}(:,1);
    end
    mu0 = mu0./numChains;
    
    V0 = Ezn_zn{1}{1}-Ez{1}(:,1)*mu0'-mu0*Ez{1}(:,1)'+mu0*mu0';
    for jj=2:numChains
        V0 = V0+Ezn_zn{jj}{1}-Ez{jj}(:,1)*mu0'-mu0*Ez{jj}(:,1)'+mu0*mu0';
    end
    V0 = V0./numChains;
    
    % update transformation matrix A
    tmp1 = zeros(size(A));
    tmp2 = zeros(size(A));
    for jj=1:numChains
        for ii=2:chainLen(jj)
            tmp1 = tmp1+Ezn_zn1{jj}{ii};
            tmp2 = tmp2+Ezn_zn{jj}{ii-1};
        end
    end
    A = tmp1/tmp2;
    
    % update gamma
    tmp = zeros(size(Gamma));
    totalCount = 0;
    for jj=1:numChains
        for ii=2:chainLen(jj)
            tmp = tmp+...
                Ezn_zn{jj}{ii}-A*Ezn_zn1{jj}{ii}'-Ezn_zn1{jj}{ii}*A'+A*Ezn_zn{jj}{ii-1}*A';
            totalCount = totalCount+1;
        end
    end
    Gamma = (1/totalCount).*tmp;
    Gamma = diag(diag(Gamma));
    
    % update C
    tmp1 = zeros(size(C));
    tmp2 = zeros(size(C));
    for jj=1:numChains
        for ii=1:chainLen(jj)
            tmp1 = tmp1+data{jj}(:,ii)*Ez{jj}(:,ii)';
            tmp2 = tmp2+Ezn_zn{jj}{ii};
        end
    end
    C = tmp1/tmp2;
    
    % update sigma
    tmp = zeros(size(Sigma));
    totalCount = 0;
    for jj=1:numChains
        for ii=1:chainLen(jj)
            tmp = tmp+data{jj}(:,ii)*data{jj}(:,ii)'...
                -C*Ez{jj}(:,ii)*data{jj}(:,ii)'-...
                data{jj}(:,ii)*Ez{jj}(:,ii)'*C'+C*Ezn_zn{jj}{ii}*C';
            totalCount = totalCount+1;
        end
    end
    Sigma = (1/totalCount)*tmp;
    
    if currentLikelihood-prevLikelihood<=tolerance
        break;
    else
        prevLikelihood = currentLikelihood;
    end
%     plot(tt,currentLikelihood,'.');hold on;pause(1/100);
end

z = Ez;
end

function [mu_n,V_n,c_n,P_n] = KalmanForwardAlgo(x,A,C,Gamma,Sigma,mu0,V0,N,K)
% KalmanForwardAlgo.m
%  run forward algorithm for Kalman filter
P_n = cell(N,1);
mu_n  = cell(N,1);
V_n = cell(N,1);
c_n = zeros(N,1);
I = eye(K);

gaussMean = C*mu0;
gaussCov = C*V0*C'+Sigma;
K = (V0*C')/gaussCov;
mu_n{1} = mu0+K*(x(:,1)-gaussMean);
V_n{1} = (I-K*C)*V0;

% sigmaDet = det(gaussCov);
c_n(1) = GetLogMvnLikelihood(x(:,1),gaussMean,gaussCov);

for ii=2:N
    P = A*V_n{ii-1}*A'+Gamma;
    gaussMean = C*A*mu_n{ii-1};
    gaussCov = C*P*C'+Sigma;
    
    K = (P*C')/gaussCov;
    mu_n{ii} = A*mu_n{ii-1}+K*(x(:,ii)-gaussMean);
    V_n{ii} = (I-K*C)*P;
    
%     sigmaDet = det(gaussCov);
    c_n(ii) = GetLogMvnLikelihood(x(:,ii),gaussMean,gaussCov);
    P_n{ii-1} = P;
end

P_n{N} = A*V_n{N}*A'+Gamma;

end

function [logPDF] = GetLogMvnLikelihood(data,mu,sigma)
logdet = 2*sum(log(diag(chol(sigma))));
logPDF = -0.5*logdet-0.5*(data-mu)'*(sigma\(data-mu));
%0.5*trace(gaussCov\(data-mu)*(data-mu)');

end

function [X] = SWEEP(X,k)
%SWEEP.m
%   Code to evaluate the SWEEP operator on row k of matrix X
%Inputs: 
%        X - a matrix, n-by-p
%        k - the row number on which to evaluate the operator, k<=n
%
%Outputs: 
%        X - the matrix after performing the SWEEP operator on row k
%
% Created by: Byron Price
% 2018/10/08

D = X(k,k);
X(k,:) = X(k,:)./D;

[n,~] = size(X);
for ii=[1:k-1,k+1:n]
   B = X(ii,k);
   X(ii,:) = X(ii,:)-B.*X(k,:);
   X(ii,k) = -B/D;
end
X(k,k) = 1/D;
end

function [muhat_n,Vhat_n,J_n] = KalmanBackwardAlgo(A,mu_n,V_n,P_n,N)
%KalmanBackwardAlgo.m
%   run backward algorithm for Kalman filter
muhat_n = cell(N,1);
Vhat_n = cell(N,1);
J_n = cell(N,1);

J_n{N} = (V_n{N}*A')/P_n{N};

muhat_n{N} = mu_n{N};
Vhat_n{N} = V_n{N};
for ii=N-1:-1:1
    J_n{ii} = (V_n{ii}*A')/P_n{ii};
    muhat_n{ii} = mu_n{ii}+J_n{ii}*(muhat_n{ii+1}-A*mu_n{ii});
    Vhat_n{ii} = V_n{ii}+J_n{ii}*(Vhat_n{ii+1}-P_n{ii})*J_n{ii}';
end
end