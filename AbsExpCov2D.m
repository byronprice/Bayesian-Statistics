function [K,Kinv,logdet,time] = AbsExpCov2D(sigsquare,rho,d)
% AbsExpCov.m
%  Code to get the absolute value exponential covariance matrix (the Matern
%   covariance for nu=1/2), along with its inverse and determinant. This
%   algorithm runs extremely quickly, because the inverse and determinant
%   are calculated in closed-form. The computation of the inverse scales
%   sublinearly with the dimensionality of the matrix.

%INPUT: sigsquare - the variance of the kernel
%       rho - the parameter rho, which determines the spread of the
%        covariance along the off-diagonal
%       d - dimensionality of the original data 2-by-1 vector
%OUTPUT: K - the covariance matrix / kernel
%        Kinv - the inverse of the covariance
%        determinant - the matrix determinant, which is used to get the
%         inverse extremely quickly.

% Created: 2018/01/19, 24 Cummington Mall, Boston, MA
%   Byron Price, PhD Candidate Boston University
% Updated: 2018/08/16
%   By: Byron Price

[K1,Kinv1,logdet1] = AbsExpCov1D(sigsquare,rho,d(1)); % dimension 1, rows
[K2,Kinv2,logdet2] = AbsExpCov1D(sigsquare,rho,d(2)); % dimension 2, columns

K = kron(K2,K1)./sigsquare;
Kinv = kron(Kinv2,Kinv1).*sigsquare;

logdet = d(2)*logdet1+d(1)*logdet2-prod(d)*log(sigsquare);

% K2 = GetMain(sigsquare,rho,d);
% 
% sum(sum(abs(K2-K)))
% 
% 2*sum(log(diag(chol(K))))-logdet
end

function [K,D] = GetMain(sigsquare,rho,d)
D = zeros(prod(d),prod(d));

leftcount = 0;
for jj=1:d(2)
    for ii=1:d(1)
        leftcount = leftcount+1;
        currentIndex = [ii,jj];
        rightcount = 0;
        for kk=1:d(2)
            for ll=1:d(1)
                rightcount = rightcount+1;
                newIndex = [ll,kk];
                D(leftcount,rightcount) = norm(currentIndex-newIndex,1);
            end
        end
    end
end
K = sigsquare.*exp(-D./rho);
end
