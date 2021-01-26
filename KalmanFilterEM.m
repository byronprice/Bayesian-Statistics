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
dataCov = data*data';
Sigma = dataCov./(N-1);


% transformation from z to x
C = normrnd(0,1,[d,K]);

% estimate of observation noise covariance
Gamma = C'*(Sigma./2)*C;

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
    [mu_n,V_n,c_n,P_n] = KalmanForwardAlgo(data,A,C,Gamma,Sigma,mu0,V0,N,K);

    if heldOut==true
        [~,~,c_n_heldout,~] = KalmanForwardAlgo(heldOutData,A,C,Gamma,Sigma,mu0,V0,M,K);
        currentLikelihood = sum(c_n_heldout);
    else
        currentLikelihood = sum(c_n);
    end
    [muhat_n,Vhat_n,J_n] = KalmanBackwardAlgo(A,mu_n,V_n,P_n,N);
    
    Ez = zeros(K,N);
    Sum_Ezn_zn = zeros(K,K);
    Sum1_Ezn_zn = zeros(K,K);
    Sum2_Ezn_zn = zeros(K,K);
    Sum_Ezn_zn1 = zeros(K,K);
    for ii=1:N
        Ez(:,ii) = muhat_n{ii};
        tmp = Vhat_n{ii}+muhat_n{ii}*muhat_n{ii}';
        Sum_Ezn_zn = Sum_Ezn_zn+tmp;
        if ii>1
            Sum_Ezn_zn1 = Sum_Ezn_zn1+J_n{ii-1}*Vhat_n{ii}+muhat_n{ii}*muhat_n{ii-1}';
            Sum2_Ezn_zn = Sum2_Ezn_zn+tmp;
        end
        if ii<N
           Sum1_Ezn_zn = Sum1_Ezn_zn+tmp;
        end
    end
    Sum_dataEz = data*Ez';
    % M step, maximize expected complete-data log likelihood
    
    % update initial conditions
    mu0 = Ez(:,1);
    V0 = Vhat_n{1};
    
    % update transformation matrix A
    A = Sum_Ezn_zn1/Sum1_Ezn_zn;
    
    % update gamma
    Gamma = (1/(N-2)).*(Sum2_Ezn_zn-A*Sum_Ezn_zn1'-Sum_Ezn_zn1*A'+...
        A*Sum1_Ezn_zn*A');
    
    % update C
    C = Sum_dataEz/Sum_Ezn_zn;
    
    % update sigma
    tmp = C*Ez*data';
    Sigma = (1/(N-1))*(dataCov-tmp-tmp'+C*Sum_Ezn_zn*C');
    
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
V0Ct = V0*C';
gaussInput = gaussCov\(x(:,1)-gaussMean);
mu_n{1} = mu0+V0Ct*gaussInput;
V_n{1} = (I-V0Ct*(gaussCov\C))*V0;

% sigmaDet = det(gaussCov);
c_n(1) = GetLogMvnLikelihood(x(:,1),gaussMean,gaussCov,gaussInput);

for ii=2:N
    P = A*V_n{ii-1}*A'+Gamma;
    oneStepPred = A*mu_n{ii-1};
    gaussMean = C*oneStepPred;
    gaussCov = C*P*C'+Sigma;
    
    PCt = P*C';
    gaussInput = gaussCov\(x(:,ii)-gaussMean);
    mu_n{ii} = oneStepPred+PCt*gaussInput;
    V_n{ii} = (I-PCt*(gaussCov\C))*P;
    
%     sigmaDet = det(gaussCov);
    c_n(ii) = GetLogMvnLikelihood(x(:,ii),gaussMean,gaussCov,gaussInput);
    P_n{ii-1} = P;
end

P_n{N} = A*V_n{N}*A'+Gamma;

end

function [logPDF] = GetLogMvnLikelihood(data,mu,sigma,sigmaInvData)
logdet = sum(log(diag(chol(sigma))));
logPDF = -logdet-0.5*(data-mu)'*sigmaInvData;
%0.5*trace(gaussCov\(data-mu)*(data-mu)');

end

function [muhat_n,Vhat_n,J_n] = KalmanBackwardAlgo(A,mu_n,V_n,P_n,N)
%KalmanBackwardAlgo.m
%   run backward algorithm for Kalman smoother
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