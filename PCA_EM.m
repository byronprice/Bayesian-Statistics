function [W,mu,sigmasquare,z,likelihood] = PCA_EM(data,K)
% PCA_EM.m
%    EM algorithm for PCA
%   C = W*W' + sigmasquare*I
%   Cinv = (1/sigmasquare)*I-(1/sigmasquare)^2*W*(I+W'*W*(1/sigmasquare))^-1*W'
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

% optimal reconstruction in squared error sense is:
%   W*((W'*W)\M*z)+mu
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
    
    MiWt = Minv*W';
    EzEzt = zeros(K,K);
    xEzt = zeros(d,K);
    xtx = 0;
    for nn=1:batchSize

        xhat = data(:,inds(nn));
        
        Ez = MiWt*xhat;
        EzEzt = EzEzt+Ez*Ez';
        xEzt = xEzt+bsxfun(@times,xhat,Ez');
        xtx = xtx-xhat'*xhat;
       
    end
    EzEzt = EzEzt+(Minv*batchSize)./precision;
    Wgrad = (xEzt-W*EzEzt).*precision;
    preGrad = xtx+d*batchSize/precision+2*sum(sum(W.*xEzt))-sum(sum(EzEzt'.*WtW));

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