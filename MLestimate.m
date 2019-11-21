function [beta,corrMat] = MLestimate(X,y)
%MLestimate.m  Project 1, 4-c
%   Calculate the ML estimator for beta, of the linear regression y =
%   X*beta ... also output the correlation matrix for beta, (X'*X)^(-1)
%     uses the QR decomposition to solve for both
%Inputs:
%        X - a design matrix, size n-by-p, where n is the number of data
%        points and p is the number of predictors (or parameters)
%        y - a response vector, size n-by-1
%Outputs:
%        beta - the maximum likelihood estimator beta
%        corrMat - the correlation matrix of beta
%
% Created by: Byron Price
% 2018/09/25

% first get Q and R (I'm assuming this part we're allowed to use simple
%   foor loops and such to get the Gram-Schmidt QR decomposition)
[n,p] = size(X);

U = zeros(n,p); % initialize U
U(:,1) = X(:,1);
for ii=2:p % iterate
   U(:,ii) = X(:,ii); % initialize u_i with a_i
   for jj=1:ii-1
      U(:,ii) = U(:,ii)-proj(X(:,ii),U(:,jj)); % sum over projections 
                          % to create orthogonal set
   end
end

Q = normalizeCols(U); % column normalize
R = crossprod(Q,X); % like the matrix C from problem 3 in this homework

% or use MATLAB:  [Q,R] = qr(X);

% force R to be upper triangular (set lower triangular to exactly 0, whereas 
%  if we avoided this step, some elements would likely be close to, but not
%  exactly, zero)
tmp = logical(tril(ones(p,p),-1));
R(tmp) = 0;

%  DO THE ML FITTING
% fit gamma
gamma = crossprod(Q,y);

% invert R
Rinv = InvUpperTri(R);

% get beta-hat, ML estimate for beta
beta = Rinv*gamma; % or use MATLAB beta = R\gamma;

% get correlation matrix for beta-hat, (X'*X)^-1
corrMat = crossprod(Rinv,Rinv,true);
end

function [Y] = crossprod(x,y,Transpose)
if nargin==2
    Transpose=false;
end
if Transpose==false
    Y = x'*y;
elseif Transpose==true
    Y = x*y'; 
end
end

function [Matrix] = normalizeCols(Matrix)
%normalizeCols.m  Project 1, 1-c
%   Normalize each column of the input Matrix by its column L2 norm
%Inputs: 
%        Matrix - an n-by-m matrix
%Outputs:
%        Matrix - the input matrix with each column normalized by its L2 norm
%
% Created by: Byron Price
% 2018/09/21
[~,m] = size(Matrix);

% iterate through each column
for ii=1:m
   L2norm = norm2(Matrix(:,ii)); % get L2 norm
   Matrix(:,ii) = Matrix(:,ii)./L2norm; % divde column by its L2 norm
end

end

