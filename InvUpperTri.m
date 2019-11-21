function [Inverse] = InvUpperTri(Matrix,Transpose)
%InvUpperTri.m  Project 1, 1-a
%   Code to take inverse of an upper triangular matrix
%Inputs: 
%        Matrix - an n-by-n upper triangular matrix
%     OPTIONAL
%        Tranpose - logical true or false, depending on whether or not you want to
%            return the inverse (false) or the inverse tranpose (true) ...
%            defaults to false
%
%Outputs: 
%        Inverse - the inverse (or inverse tranpose) of Matrix
%
% Created by: Byron Price
% 2018/09/21

if istriu(Matrix)==false
    fprintf('Matrix must be upper triangular\n');
    Inverse = NaN;
    return;
end

if nargin==1
    Transpose = false;
end

[n,~] = size(Matrix);

Inverse = [Matrix,eye(n)]; % create extended matrix with identity to the right

% loop through each row and run adjust operator
for ii=n:-1:1
    Inverse = Adjust(Inverse,ii);
end

Inverse = Inverse(:,n+1:end); % left side of extended matrix is now identity, 
                         % right side is the inverse

if Transpose==true
    Inverse = Inverse';
end

end

function [A] = Adjust(A,k)
% Adjust operator, uses the k-th row of A as way to eliminate values in the
% other rows of its left-hand side

A(k,:) = A(k,:)./A(k,k);

for ii=1:k-1
   A(ii,:) = A(ii,:)-A(ii,k).*A(k,:);
end

% A(1:k-1,:) = A(1:k-1,:)-A(1:k-1,k)*A(k,:); % matrix operation to perform
                              % same function as for loop above

end

