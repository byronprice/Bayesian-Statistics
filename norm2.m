function [L2Norm] = norm2(vector)
%norm2.m  Project 1, 1-b
%  Take the L2 norm of a vector (square root of the sum of squared
%       elements)
%Inputs:
%       vector - an n-by-1 traditional (column) vector
%Outputs: 
%       L2norm - the L2 norm of the vector
% 
% Created by: Byron Price
% 2018/09/21

vector = vector(:); % force to be a column vector
maxVal = max(abs(vector)); 
vector = vector./maxVal; % divide out maximum value for numerical stability
L2Norm = maxVal*sqrt(vector'*vector); % take L2norm

end