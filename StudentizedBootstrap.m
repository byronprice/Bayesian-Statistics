function [CI,CIpivotal] = StudentizedBootstrap(data,func,iters)
% StudentizedBootstrap.m
%   compute a confidence interval from the studentized bootstrap
%      best for pivotal statistics

N = length(data);
alpha = 0.05;

if nargin<2
    func = @(x) mean(x);
    iters = 1000;
elseif nargin<3
    iters = 1000;
end

thetaHat = func(data);

bootTheta = zeros(iters,1);
bootT = zeros(iters,1);
for ii=1:iters
    bootData = data(ceil(rand([N,1])*N));
    bootTheta(ii) = func(bootData);
    
    bootTheta2 = zeros(iters,1);
    for jj=1:iters
        bootData2 = bootData(ceil(rand([N,1])*N));
        bootTheta2(jj) = func(bootData2);
    end
    bootT(ii) = (bootTheta(ii)-thetaHat)/std(bootTheta2);
end

sd = std(bootTheta);

CI = [thetaHat-sd*quantile(bootT,1-alpha/2),...
    thetaHat-sd*quantile(bootT,alpha/2)];
CIpivotal = [2*thetaHat-quantile(bootTheta,1-alpha/2),...
    2*thetaHat-quantile(bootTheta,alpha/2)];
end