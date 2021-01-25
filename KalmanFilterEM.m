function [A,C,Gamma,Sigma,z,mu0,V0] = KalmanFilterEM(data,K,heldOutData)
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
%INPUT: data - observed data, input as a matrix d-by-N, where N is the
%        number of observations in the chain and d is the dimensionality 
%        of each observation
%  OPTIONAL
%       K - dimensionality of latent space
%       heldOutData - observed data, d-by-M, which is held-out for
%         cross-validation purposes, to avoid overfitting
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

[d,N] = size(data);

if nargin==1
    heldOut = false;
    K = d;
elseif nargin==2
    heldOut = false;
else
    heldOut = true;
    [~,M] = size(heldOutData);
end

% initialize parameters
Sigma = cov(data');

% estimate of observation noise covariance
if K==d
    Gamma = Sigma./2;
else
    Gamma = normrnd(0,1,[K,K]);
    Gamma = Gamma'*Gamma;
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
% A = mldivide(data(:,1:end-1)',data(:,2:end)')';

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
prevLikelihood = -1e10;
for tt=1:maxIter
    % E step, get expected hidden state estimates
    [mu_n,V_n,c_n,P_n] = KalmanForwardAlgo(data,A,C,Gamma,Sigma,mu0,V0,N,d);

    if heldOut==true
        [~,~,c_n_heldout,~] = KalmanForwardAlgo(heldOutData,A,C,Gamma,Sigma,mu0,V0,M,d);
        currentLikelihood = sum(c_n_heldout);
    else
        currentLikelihood = sum(c_n);
    end
    [muhat_n,Vhat_n,J_n] = KalmanBackwardAlgo(A,mu_n,V_n,P_n,N);
    
    Ez = zeros(d,N);
    Ezn_zn = cell(N,1);
    Ezn_zn1 = cell(N,1);
%     Ezn1_zn = cell(N,1);
    for ii=1:N
        Ez(:,ii) = muhat_n{ii};
        if ii>1
            Ezn_zn1{ii} = J_n{ii-1}*Vhat_n{ii}+muhat_n{ii}*muhat_n{ii-1}';
%             Ezn1_zn{ii} = J_n{ii}*Vhat_n{ii-1}+muhat_n{ii-1}*muhat_n{ii}';
        end
        Ezn_zn{ii} = Vhat_n{ii}+muhat_n{ii}*muhat_n{ii}';
    end
    
    % M step, maximize expected log-likelihood
    
    % update initial conditions
    mu0 = Ez(:,1);
    V0 = Vhat_n{1};
    
    % update transformation matrix A
    tmp1 = zeros(size(A));
    tmp2 = zeros(size(A));
    for ii=2:N
        tmp1 = tmp1+Ezn_zn1{ii};
        tmp2 = tmp2+Ezn_zn{ii-1};
    end
    A = tmp1/tmp2;
    
    % update gamma
    tmp = zeros(size(Gamma));
    for ii=2:N
        tmp = tmp+...
           Ezn_zn{ii}-A*Ezn_zn1{ii}'-Ezn_zn1{ii}*A'+A*Ezn_zn{ii-1}*A';
    end
    Gamma = (1/(N-1)).*tmp;
    
    % update C
    tmp1 = zeros(size(C));
    tmp2 = zeros(size(C));
    for ii=1:N
       tmp1 = tmp1+data(:,ii)*Ez(:,ii)';
       tmp2 = tmp2+Ezn_zn{ii};
    end
    C = tmp1/tmp2;
    
    % update sigma
    tmp = zeros(size(Sigma));
    for ii=1:N
        tmp = tmp+data(:,ii)*data(:,ii)'...
        -C*Ez(:,ii)*data(:,ii)'-...
            data(:,ii)*Ez(:,ii)'*C'+C*Ezn_zn{ii}*C';
    end
    Sigma = (1/N)*tmp;
    
    if currentLikelihood-prevLikelihood<=tolerance
        break;
    else
        prevLikelihood = currentLikelihood;
    end
%     plot(tt,currentLikelihood,'.');hold on;pause(1/100);
end

z = Ez;
end

function [mu_n,V_n,c_n,P_n] = KalmanForwardAlgo(x,A,C,Gamma,Sigma,mu0,V0,N,d)
% KalmanForwardAlgo.m
%  run forward algorithm for Kalman filter
P_n = cell(N,1);
mu_n  = cell(N,1);
V_n = cell(N,1);
c_n = zeros(N,1);
I = eye(d);

gaussMean = C*mu0;
gaussCov = C*V0*C'+Sigma;
K = (V0*C')/gaussCov;
mu_n{1} = mu0+K*(x(:,1)-gaussMean);
V_n{1} = (I-K*C)*V0;

% sigmaInv = gaussCov;sigmaDet = det(gaussCov);
% for jj=1:d
%     sigmaInv = SWEEP(sigmaInv,jj);
% end

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