function [W,mu,Psi,z,likelihood] = FactorAnalysis_EM(data,K)
% FactorAnalysis_EM.m
%    EM algorithm for latent factor analysis

[d,N] = size(data);
[U,~,~] = svd(data,'econ');

W = U(:,1:K);
Ik = eye(K);

mu = mean(data,2);

data = data-repmat(mu,[1,N]);

dataCov = sum(data.*data,2)./N;

Psi = dataCov;

maxIter = 1000;
tolerance = 1e-6;oldW = W;
for iter=1:maxIter
    % E-step expectations
    WtPsiInv = W'.*(repmat(1./Psi',[K,1]));
    G = (Ik+WtPsiInv*W)\Ik;
    GWtPsiInv = (Ik+WtPsiInv*W)\WtPsiInv;
    
    Sum_dataEz = zeros(d,K);
    Sum_EzEz = zeros(K,K);
    
    for nn=1:N
        Ez = GWtPsiInv*data(:,nn);
        Sum_dataEz = Sum_dataEz+data(:,nn)*Ez';
        EzEzt = Ez*Ez';
        Sum_EzEz = Sum_EzEz+EzEzt;
        
    end
    Sum_EzEz = Sum_EzEz+N*G;
    
    W = Sum_dataEz/Sum_EzEz;
    Sum_dataEz = Sum_dataEz/N;
    Psi = dataCov-sum(W.*Sum_dataEz,2);
    
    difference = norm(W-oldW)./norm(oldW);
    if difference<tolerance
        break;
    end
    oldW = W;
end

WtPsiInv = W'.*(repmat(1./Psi',[K,1]));
G = (Ik+WtPsiInv*W)\Ik;

z = zeros(K,N);
for nn=1:N
    z(:,nn) = G*WtPsiInv*data(:,nn);
end
[likelihood] = GetLikelihood(data,W,Psi,N,d);
end

function [loglikelihood] = GetLikelihood(data,W,Psi,N,d)

Cinv = (W*W'+diag(Psi))\eye(d);
loglikelihood = N*sum(log(diag(chol(Cinv))));
for nn=1:N
    loglikelihood = loglikelihood-0.5*data(:,nn)'*Cinv*data(:,nn);
end
end