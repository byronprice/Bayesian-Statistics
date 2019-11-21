function [Q,R] = QR_Decomp(X)
%QR_Decomp.m  Project 1, 4-e
%   Calculate the QR decomposition of the matrix, X=QR, as long as X satisfies
%   two conditions: 1) X is tall (n-by-p) with n>=p; 2) X is full rank
%   (columns are linearly independent) ... this ensures that R will be the
%   Cholesky factor of X'X
%         R'R = X'X and X = QR
%     uses Gram-Schmidt orthogonalization (which appears to be numerically
%     unstable, as the columns of Q are not always perfectly orthogonal)
%Inputs:
%        X - a matrix, size n-by-p, where n>=p and X full rank (the columns
%        of X are linearly independent)
%Outputs:
%        Q - n-by-n orthonormal matrix, of QR decomposition
%        R - p-by-p upper triangular matrix, which is also the Cholesky factor
%         for the matrix X'X (X-transpose X)
%
%
% Created by: Byron Price
% 2018/09/25

% check that conditions are met
[n,p] = size(X);

matRank = rank(X);

if n>=p && matRank==p
    % conditions are met ... calculate Q and R
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
    
    % force R to be upper triangular (set lower triangular to exactly 0, whereas
    %  if we avoided this step, some elements would likely be close to, but not
    %  exactly, zero)
    tmp = logical(tril(ones(p,p),-1));
    R(tmp) = 0;
else
   Q = NaN;R = NaN; 
end

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