function [K,Kinv,logdeterminant,cholInv] = AbsExpCov1D(sigsquare,rho,d)
% AbsExpCov.m
%  Code to get the absolute value exponential covariance matrix (the Matern
%   covariance for nu=1/2), along with its inverse and determinant. This
%   algorithm runs extremely quickly, because the inverse and determinant
%   are calculated in closed-form. The computation of the inverse scales
%   sublinearly with the dimensionality of the matrix.

%INPUT: sigsquare - the variance of the kernel
%       rho - the parameter rho, which determines the spread of the
%        covariance along the off-diagonal
%       d - dimensionality of the kernel
%OUTPUT: K - the covariance matrix / kernel
%        Kinv - the inverse of the covariance
%        logdeterminant - the log matrix determinant, which is used to get the
%         inverse extremely quickly.
%        cholInv - Cholesky decomposition of the inverse

% Created: 2018/01/19, 24 Cummington Mall, Boston, MA
%   Byron Price, PhD Candidate Boston University
% Updated: 2019/12/17
%   By: Byron Price

[K] = GetMain(sigsquare,rho,d);
[logdeterminant,~] = GetDet(sigsquare,rho,d);

[Kinv] = GetInverse(sigsquare,rho,d);

[cholInv] = GetCholeskyInv(sigsquare,rho,d);
end

function [K,D] = GetMain(sigsquare,rho,d)
D = zeros(d,d); % along 1D space, what about 2D?
for ii=1:d
    for jj=1:d
        D(ii,jj) = abs(ii-jj);
    end
end
K = sigsquare.*exp(-D./rho);
end

function [logdeterminant,ratio] = GetDet(sigsquare,rho,d)
if d==2
%    determinant = (sigsquare.^2).*(1-exp(-1./rho).^2);
    logdeterminant = 2*log(sigsquare)+log(1-exp(-2/rho));
    
%     [K,~] = GetMain(sigsquare,rho,2);
%     logdeterminant = 2*sum(log(diag(chol(K))));
    ratio = log(sigsquare)+log(1-exp(-2/rho));
else
    [det2,ratio] = GetDet(sigsquare,rho,2);
%     ratio = log(sigsquare)+log(1-exp(-2/rho));
    logdeterminant = ratio*(d-2)+det2;
end
end

function [inverse] = GetInverse(sigsquare,rho,d)

inverse = sparse(d,d);oneVec = ones(d-1,1);
mainDiagInds = find(eye(d));

[det3,ratio] = GetDet(sigsquare,rho,3);

inverse(1,1) = exp(-ratio);inverse(d,d) = exp(-ratio);
mainDiagInds = mainDiagInds(2:end-1);

% inverse(mainDiagInds) = ((sigsquare.^2).*(1-exp(-2./rho).^2))./det3;
inverse(mainDiagInds) = exp(2*log(sigsquare)+log(1-exp(-4/rho))-det3);

temp = ((sigsquare.^2).*(exp(-1./rho)).*(exp(-2./rho)-1))./exp(det3);
% temp = exp(2*log(sigsquare)-2/rho-det3);

inverse(logical(diag(oneVec,1))) = temp;
inverse(logical(diag(oneVec,-1))) = temp;
end

function [cholInv] = GetCholeskyInv(sigsquare,rho,d)
cholInv = sparse(d,d);oneVec = ones(d-1,1);

[det3,ratio] = GetDet(sigsquare,rho,3);

cholInv(logical(eye(d))) = sqrt(exp(-ratio));

invDiag = exp(2*log(sigsquare)+log(1-exp(-4/rho))-det3);

cholInv(logical(diag(oneVec,1))) = -sqrt(invDiag-exp(-ratio));

cholInv(d,d) = sqrt(2*exp(-ratio)-invDiag);
end