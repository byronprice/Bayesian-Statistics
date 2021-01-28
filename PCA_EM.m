function [W,mu,sigmasquare,z,likelihood] = PCA_EM(data,K)
% PCA_EM.m
%    EM algorithm for PCA

[d,N] = size(data);

W = normrnd(0,1,[d,K]);
sigmasquare = var(data(:));
Ik = eye(K);

mu = mean(data,2);

data = data-repmat(mu,[1,N]);

dataVar = sum(sum(data.*data));

maxIter = 1000;
tolerance = 1e-6;oldW = W;
for iter=1:maxIter
    % E-step expectations
    Minv = (W'*W+Ik.*sigmasquare)\Ik;
    
    Sum_dataEz = zeros(d,K);
    Sum_EzEz = zeros(K,K);
    Sum_sigmasquare = dataVar;
    
    WtW = W'*W;
    for nn=1:N
        tmp = W'*data(:,nn);
        Ez = Minv*tmp;
        Sum_dataEz = Sum_dataEz+data(:,nn)*Ez';
        EzEzt = Ez*Ez';
        Sum_EzEz = Sum_EzEz+EzEzt;
        
        Sum_sigmasquare = Sum_sigmasquare-2*Ez'*tmp+sum(sum(EzEzt'.*WtW));
    end
    Sum_EzEz = Sum_EzEz+N*sigmasquare*Minv;
    
    W = Sum_dataEz/Sum_EzEz;
    sigmasquare = (1/(N*d))*Sum_sigmasquare;
    
    difference = norm(W-oldW)./norm(oldW);
    if difference<tolerance
        break;
    end
    oldW = W;
end

Minv = (W'*W+Ik.*sigmasquare)\Ik;
z = zeros(K,N);
for nn=1:N
    z(:,nn) = Minv*W'*data(:,nn);
end

[likelihood] = GetLikelihood(data,W,sigmasquare,N,d);
end

% online version
function [W,sigmasquare] = OnlinePCA_EM(data,K)
[d,N] = size(data);

W = normrnd(0,1,[d,K]);
sigmasquare = var(data(:));
Ik = eye(K);

nu = 1e-2;

maxIter = 1e3;
batchSize = 10;
for iter=1:maxIter
    inds = randperm(N,batchSize);
    
    WtW = W'*W;
    Minv = (WtW+Ik.*sigmasquare)\Ik;
    
    Wgrad = zeros(size(W));
    sigmaGrad = 0;
    for nn=1:batchSize
        
        
        xhat = data(:,inds(nn));
        
        Wtx = W'*xhat;
        Ez = Minv*Wtx;
        EzEzt = sigmasquare*Minv+Ez*Ez';
        
        Wgrad = Wgrad+(xhat*Ez'-W*EzEzt);
        
        sigmaGrad = sigmaGrad+(-d+(xhat'*xhat)/sigmasquare-2*Ez'*Wtx/sigmasquare+...
            sum(sum(EzEzt'.*WtW))/sigmasquare);
    end
    W = W+nu*Wgrad./batchSize;
    sigmasquare = max(sigmasquare+nu*sigmaGrad./batchSize,1e-9);
end

end

function [loglikelihood] = GetLikelihood(data,W,sigmasquare,N,d)

Cinv = (W*W'+eye(d)*sigmasquare)\eye(d);
loglikelihood = N*sum(log(diag(chol(Cinv))));
for nn=1:N
    loglikelihood = loglikelihood-0.5*data(:,nn)'*Cinv*data(:,nn);
end
end