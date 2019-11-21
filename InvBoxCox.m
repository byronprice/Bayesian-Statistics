function [y] = InvBoxCox(w,lambda)
%BoxCox.m
%   box cox transformation

tolerance = 1e-9;
if abs(lambda)<=tolerance
    y = exp(w);
else
    y = exp(log(w.*lambda+1)./lambda);
end
end