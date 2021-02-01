function [W,mu,sigmasquare,z,likelihood] = PCA_EM(data,K)
% PCA_EM.m
%    EM algorithm for PCA

[d,N] = size(data);

mu = mean(data,2);

data = data-repmat(mu,[1,N]);

dataVar = sum(sum(data.*data));

W = normrnd(0,1,[d,K]);
sigmasquare = var(data(:));
Ik = eye(K);

maxIter = 1000;
tolerance = 1e-6;oldW = W;
for iter=1:maxIter
    % E-step expectations
    Minv = (W'*W+Ik.*sigmasquare)\Ik;
    
    Sum_dataEz = zeros(d,K);
    Sum_EzEz = zeros(K,K);
    Sum_sigmasquare = dataVar;
    
    WtW = W'*W;
    sigSquareMinv = sigmasquare*Minv;
    for nn=1:N
        Wtx = W'*data(:,nn);
        Ez = Minv*Wtx;
        Sum_dataEz = Sum_dataEz+data(:,nn)*Ez';
        EzEzt = sigSquareMinv+Ez*Ez';
        Sum_EzEz = Sum_EzEz+EzEzt;
        
        Sum_sigmasquare = Sum_sigmasquare-2*Ez'*Wtx+sum(sum(EzEzt'.*WtW));
    end
    
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
precision = 1./var(data(:));
Ik = eye(K);

nu = 1e-2;

maxIter = 5e3;
batchSize = 10;
for iter=1:maxIter
    inds = randperm(N,batchSize);
    
    WtW = W'*W;
    Minv = (WtW+Ik./precision)\Ik;
    
    Wgrad = zeros(size(W));
    preGrad = 0;
    for nn=1:batchSize

        xhat = data(:,inds(nn));
        
        Wtx = W'*xhat;
        Ez = Minv*Wtx;
        EzEzt = Minv./precision+Ez*Ez';
        
        Wgrad = Wgrad+(xhat*Ez'-W*EzEzt);
        
        preGrad = preGrad+(d/precision-(xhat'*xhat)+2*Ez'*Wtx-...
            sum(sum(EzEzt'.*WtW)));
    end
    Wgrad = Wgrad.*precision;
    W = W+nu*Wgrad./batchSize;
    precision = max(precision+nu*preGrad./batchSize,1e-9);
end
sigmasquare = 1/precision;
end

function [loglikelihood] = GetLikelihood(data,W,sigmasquare,N,d)

Cinv = (W*W'+eye(d)*sigmasquare)\eye(d);
loglikelihood = N*sum(log(diag(chol(Cinv))));
for nn=1:N
    loglikelihood = loglikelihood-0.5*data(:,nn)'*Cinv*data(:,nn);
end
end