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
dataCov = zeros(d,d);
totalCount = 0;
for jj=1:numChains
    for ii=1:chainLen(jj)
        dataCov = dataCov+data{jj}(:,ii)*data{jj}(:,ii)';
        totalCount = totalCount+1;
    end
end
Sigma = (1/(totalCount-1))*dataCov;

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

if K==d
    mu0 = mean(data{1},2);
    V0 = Sigma;
else
    mu0 = zeros(K,1);
    V0 = Gamma;
end


maxIter = 5e3;
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
    SigmaInv = Sigma\eye(d);
    for ii=1:numChains
        [mu_n{ii},V_n{ii},c_n{ii},P_n{ii}] = KalmanForwardAlgo(data{ii},A,C,Gamma,SigmaInv,mu0,V0,chainLen(ii),K);
    end

    if heldOut==true
        currentLikelihood = 0;
        for ii=1:numHeldOut
            [~,~,c_n_heldout,~] = KalmanForwardAlgo(heldOutData{ii},A,C,Gamma,SigmaInv,mu0,V0,heldOutLen(ii),K);
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
    
    % calculate expectations and sufficient statistics of the expectations
    Ez = cell(numChains,1);
    Ezn_zn_first = cell(numChains,1);
    
    Sum_dataEz = zeros(d,K);
    Sum_Ezn_zn = zeros(K,K);
    Sum1_Ezn_zn = zeros(K,K);
    Sum2_Ezn_zn = zeros(K,K);
    Sum_Ezn_zn1 = zeros(K,K);

    for jj=1:numChains
        Ez{jj} = zeros(K,chainLen(jj));
        for ii=1:chainLen(jj)
            Ez{jj}(:,ii) = muhat_n{jj}{ii};
            tmp = Vhat_n{jj}{ii}+muhat_n{jj}{ii}*muhat_n{jj}{ii}';
            Sum_Ezn_zn = Sum_Ezn_zn+tmp;
            
            if ii==1
                Ezn_zn_first{jj} = tmp;
            end
            if ii>1
                Sum_Ezn_zn1 = Sum_Ezn_zn1+J_n{jj}{ii-1}*Vhat_n{jj}{ii}+muhat_n{jj}{ii}*muhat_n{jj}{ii-1}';
                Sum2_Ezn_zn = Sum2_Ezn_zn+tmp;
            end
            if ii<chainLen(jj)
                Sum1_Ezn_zn = Sum1_Ezn_zn+tmp;
            end
        end
        Sum_dataEz = Sum_dataEz+data{jj}*Ez{jj}';
    end
    
    % M step, maximize expected complete-data log likelihood
    
    % update initial conditions
    mu0 = Ez{1}(:,1);
    for jj=2:numChains
        mu0 = mu0+Ez{jj}(:,1);
    end
    mu0 = mu0./numChains;
    
    V0 = Ezn_zn_first{1}-Ez{1}(:,1)*mu0'-mu0*Ez{1}(:,1)'+mu0*mu0';
    for jj=2:numChains
        V0 = V0+Ezn_zn_first{jj}-Ez{jj}(:,1)*mu0'-mu0*Ez{jj}(:,1)'+mu0*mu0';
    end
    V0 = V0./numChains;
    
    % update transformation matrix A
    A = Sum_Ezn_zn1/Sum1_Ezn_zn;
    
    % update gamma
    totalCount = sum(chainLen-1)-numChains; % unbiased estimator
    Gamma = (1/totalCount).*(Sum2_Ezn_zn-A*Sum_Ezn_zn1'-Sum_Ezn_zn1*A'+...
        A*Sum1_Ezn_zn*A');
%     Gamma = diag(Gamma);
    
    % update C
    C = Sum_dataEz/Sum_Ezn_zn;
    
    % update sigma
    totalCount = sum(chainLen)-numChains; % unbiased estimator
    tmp = C*Sum_dataEz';
    Sigma = (1/totalCount)*(dataCov-tmp-tmp'+C*Sum_Ezn_zn*C');
    
    if currentLikelihood-prevLikelihood<=tolerance
        break;
    else
        prevLikelihood = currentLikelihood;
    end
%    plot(tt,currentLikelihood,'.');hold on;pause(1/100);
end

z = Ez;
end

function [mu_n,V_n,c_n,P_n] = KalmanForwardAlgo(x,A,C,Gamma,SigmaInv,mu0,V0,N,K)
% KalmanForwardAlgo.m
%  run forward algorithm for Kalman filter
P_n = cell(N,1);
mu_n  = cell(N,1);
V_n = cell(N,1);
c_n = zeros(N,1);
I = eye(K);

gaussMean = C*mu0;
% gaussCov = C*V0*C'+Sigma;
gaussInv = SigmaInv-SigmaInv*C*((V0\I+C'*SigmaInv*C)\C'*SigmaInv);
V0Ct = V0*C';
gaussInput = gaussInv*(x(:,1)-gaussMean);
mu_n{1} = mu0+V0Ct*gaussInput;
V_n{1} = (I-V0Ct*gaussInv*C)*V0;

c_n(1) = GetLogMvnLikelihood(x(:,1),gaussMean,gaussInv,gaussInput);

for ii=2:N
    P = A*V_n{ii-1}*A'+Gamma;
    oneStepPred = A*mu_n{ii-1};
    gaussMean = C*oneStepPred;
%     gaussCov = C*P*C'+Sigma;
    gaussInv = SigmaInv-SigmaInv*C*((P\I+C'*SigmaInv*C)\C'*SigmaInv);
    
    PCt = P*C';
    gaussInput = gaussInv*(x(:,ii)-gaussMean);
    mu_n{ii} = oneStepPred+PCt*gaussInput;
    V_n{ii} = (I-PCt*gaussInv*C)*P;
    
%     sigmaDet = det(gaussCov);
    c_n(ii) = GetLogMvnLikelihood(x(:,ii),gaussMean,gaussInv,gaussInput);
    P_n{ii-1} = P;
end

P_n{N} = A*V_n{N}*A'+Gamma;

end

function [logPDF] = GetLogMvnLikelihood(data,mu,sigmaInv,sigmaInvData)
logdet = sum(log(diag(chol(sigmaInv))));
logPDF = logdet-0.5*(data-mu)'*sigmaInvData;
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