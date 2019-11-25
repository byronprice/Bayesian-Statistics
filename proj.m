function [projection] = proj(a,u)
%proj.m  Project 1, 1-d
%   the projection of the vector a into the vector u
%
%Inputs:
%        a - an n-by-1 traditional (column) vector
%        u - another n-by-1 vector
%Outputs:
%        projection - the projection of a into u
%
% Created by: Byron Price
% 2018/09/21
a = a(:);u = u(:); % guarantee we have column vector
L2norm = norm2(u);u = u./L2norm; % divide out L2 norm for numerical stability
projection = (u'*a)*u; % take projection
end