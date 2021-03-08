function [basis,knots,beta,N] = NaturalCubicSplines(x,knots,y)
%  compute  natural spline basis for a given set of points x and knot
%   locations in knots
%     not very numerically stable, so not recommendable when abs(x)>~100

if nargin<2
    knots = quantile(x,0.1:0.2:0.9);
elseif nargin<3
    getY = false;
    beta = NaN;
    N = NaN;
else
    getY = true;
end
K = length(knots);

basis = zeros(length(x),K);
basis(:,1) = ones(length(x),1);
basis(:,2) = x;
for kk=1:K-2
    basis(:,kk+2) = (knots(K)-knots(kk))*...
        (GetDkX(x,knots(kk),knots(K))-GetDkX(x,knots(K-1),knots(K)));
end

if getY
    % smoothing splines, with knots at every unique point in x
    [N,beta] = RegressSmoothSplines(x,y);
end
end

function [dkx] = GetDkX(X,psik,psiK)
dkx = max((X-psik),0).^3-max((X-psiK),0).^3;
dkx = dkx./(psiK-psik);

end

function [N,Omega] = GetOmega(X)
knots = unique(sort(X));
K = length(knots);

N = zeros(length(X),K);
N(:,1) = 1;
N(:,2) = X;

for kk=1:K-2
    N(:,kk+2) = (knots(K)-knots(kk))*...
        (GetDkX(X,knots(kk),knots(K))-GetDkX(X,knots(K-1),knots(K)));
end

Omega = zeros(K,K);maxx = max(x);
k1tok1int = GetSplineIntegral(maxx,knots,K-1,K-1,K);
for jj=1:K-2
    for kk=1:K-2
        jtokint = GetSplineIntegral(maxx,knots,jj,kk,K);
        jtok1int = GetSplineIntegral(maxx,knots,jj,K-1,K);
        ktok1int = GetSplineIntegral(maxx,knots,kk,K-1,K);
        Omega(jj+2,kk+2) = jtokint-jtok1int-ktok1int+k1tok1int;    
    end
end

end

function [intDkxDkj] = GetSplineIntegral(maxx,knots,j,k,K)
whichKnot = max(knots(j),knots(k));

intfun = @(x,psik,psij) ((1/3)*x.^3-(psik/2)*x.^2-(psij/2)*x.^2+psik*psij*x);

intDkxDkj = intfun(maxx,knots(j),knots(k))-intfun(whichKnot,knots(j),knots(k));
intDkxDkj = intDkxDkj+intfun(maxx,knots(k),knots(K))-intfun(knots(K),knots(k),knots(K));
intDkxDkj = intDkxDkj+intfun(maxx,knots(j),knots(K))-intfun(knots(K),knots(j),knots(K));
intDkxDkj = intDkxDkj+intfun(maxx,knots(K),knots(K))-intfun(knots(K),knots(K),knots(K));
intDkxDkj = 36*intDkxDkj;

end

function [N,beta] = RegressSmoothSplines(X,Y)
[N,Omega] = GetOmega(X);

lambda = 10;
% M = N'*N+lambda*Omga;
% L = chol(M,'lower'); % M = L*L';
beta = (N'*N+lambda*Omega)\(N'*Y);

% figure;subplot(1,2,1);plot(X,N*beta,'.');
% subplot(1,2,2);plot(X,Y,'.');

% S = N*((N'*N+lambda*Omega)\N');
end