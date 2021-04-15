function [basis,knots,theta,N,Omega] = NaturalCubicSplines(x,knots,y)
%  compute  natural spline basis for a given set of points x and knot
%   locations in knots
%     not very numerically stable, so not recommendable when abs(x)>~100

if nargin<2
    knots = quantile(x,1/6:1/6:1-1/6);
    getY = false;
elseif nargin<3
    getY = false;
    theta = NaN;
    N = NaN;
    Omega = NaN;
else
    getY = true;
end

K = length(knots);

basis = GetNaturalSplines(x,K,knots);

if getY
    % smoothing splines, with knots at every unique point in x
    lambda = 1;
    [N,Omega,theta] = RegressSmoothSplines(x,y,lambda);
end
end

function [basis] = GetNaturalSplines(x,K,knots)
basis = zeros(length(x),K);
basis(:,1) = ones(length(x),1);
basis(:,2) = x;
for kk=1:K-2
    basis(:,kk+2) = GetDkX(x,knots(kk),knots(K))-GetDkX(x,knots(K-1),knots(K));
end

end

function [dkx] = GetDkX(X,psik,psiK)
dkx = max((X-psik),0).^3-max((X-psiK),0).^3;
dkx = dkx./(psiK-psik);

end

function [N,Omega,theta] = RegressSmoothSplines(X,Y,lambda)
[N,Omega] = GetOmega(X);

% M = N'*N+lambda*Omga;
% L = chol(M,'lower'); % M = L*L';
theta = (N'*N+lambda*Omega)\(N'*Y);

% figure;subplot(1,2,1);plot(X,N*beta,'.');
% subplot(1,2,2);plot(X,Y,'.');

% S = N*((N'*N+lambda*Omega)\N');
end

function [N,Omega] = GetOmega(X)
knots = unique(sort(X));
K = length(knots);

N = GetNaturalSplines(X,K,knots);

Omega = zeros(K,K);
for jj=1:K-2
    for kk=1:K-2
        Omega(jj+2,kk+2) = GetSplineIntegral(knots,jj,kk,K);  
    end
end

end

function [intDkxDkj] = GetSplineIntegral(knots,j,k,K)
whichKnot = max(knots(j),knots(k));
knotK = knots(K);
knotK1 = knots(K-1);

intfun = @(x,psik,psij) ((1/3)*x.^3-(psik/2)*x.^2-(psij/2)*x.^2+psik*psij*x);

intDkxDkj = intfun(knotK1,knots(j),knots(k))-intfun(whichKnot,knots(j),knots(k));

intDkxDkj2 = intfun(knotK,knotK,knotK)-intfun(knotK1,knotK,knotK);
intDkxDkj2 = intDkxDkj2*((knots(j)-knotK1)*(knots(k)-knotK1))/...
    ((knotK-knotK1)*(knotK-knotK1));

intDkxDkj = (36*(intDkxDkj+intDkxDkj2))/((knotK-knots(j))*(knotK-knots(k)));

end

