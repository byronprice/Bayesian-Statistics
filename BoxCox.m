function [w] = BoxCox(y,lambda)
%BoxCox.m
%   box cox transformation

tolerance = 1e-9;
if abs(lambda)<=tolerance
    w = log(y);
else
    w = ((y.^lambda)-1)./lambda;
end
end

