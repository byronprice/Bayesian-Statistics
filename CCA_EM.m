function [W,mu,Psi,z,likelihood] = CCA_EM(data,K)
% CCA_EM.m
%    group factor analysis (aka canonical correlations analysis)
%    data should be a cell array (groups by 1), each group being
%      size d-by-T (d can vary across groups but T cannot)
%    K is the number of latent dimensions

G = length(data); % number of groups

d = 0;
T = size(data{1},2);
U = cell(G,1);
W = [];
mu = [];
diagCov = [];
Psi = [];
newData = [];

for gg=1:G
    [dd,~] = size(data{gg});
    d = d+dd;
    [U{gg},~,~] = svd(data{gg},'econ');

    W = [W;U{gg}(:,1:K)];
    currMu = mean(data{gg},2);
    mu = [mu;currMu];
    data{gg} = data{gg}-repmat(currMu,[1,T]);
    tmpCov = sum(data{gg}.*data{gg},2)./(T-1);
    diagCov = [diagCov;tmpCov];
    Psi = [Psi;tmpCov];
    newData = [newData;data{gg}];
end
Ik = eye(K);

maxIter = 1000;
tolerance = 1e-6;oldW = W;
for iter=1:maxIter
    % E-step expectations
    WtPsiInv = W'.*(repmat(1./Psi',[K,1]));
    G = (Ik+WtPsiInv*W)\Ik;
    GWtPsiInv = (Ik+WtPsiInv*W)\WtPsiInv;
    
    Sum_dataEz = zeros(d,K);
    Sum_EzEz = zeros(K,K);
    
    for tt=1:T
        Ez = GWtPsiInv*newData(:,tt);
        Sum_dataEz = Sum_dataEz+newData(:,tt)*Ez';
        EzEzt = Ez*Ez';
        Sum_EzEz = Sum_EzEz+EzEzt;
        
    end
    Sum_EzEz = Sum_EzEz+T*G;
    
    W = Sum_dataEz/Sum_EzEz;
    Sum_dataEz = Sum_dataEz/T;
    Psi = diagCov-sum(W.*Sum_dataEz,2);
    
    difference = norm(W-oldW)./norm(oldW);
    if difference<tolerance
        break;
    end
    oldW = W;
end

WtPsiInv = W'.*(repmat(1./Psi',[K,1]));
G = (Ik+WtPsiInv*W)\Ik;

z = zeros(K,T);
for tt=1:T
    z(:,tt) = G*WtPsiInv*newData(:,tt);
end
[likelihood] = GetLikelihood(newData,W,Psi,T,d);
end

function [loglikelihood] = GetLikelihood(data,W,Psi,T,d)

Cinv = (W*W'+diag(Psi))\eye(d);
loglikelihood = T*sum(log(diag(chol(Cinv))));
for nn=1:T
    loglikelihood = loglikelihood-0.5*data(:,nn)'*Cinv*data(:,nn);
end
end