% MarkovChain_MonteCarlo.m

%  Metropolis-Hastings
% gaussian proposal distribution N(x,100)
%  actual distribution up to a normalization factor
%  p(x) ~ 0.3exp(-0.2*x*x)+0.7exp(-0.2*(x-10)*(x-10))

N = 100000;
sigma = 100;

x = zeros(N,1);
x(1) = normrnd(0,sigma);

for ii=2:N
    u = rand;
    tempX = x(ii-1)+normrnd(0,1);
%     tempX = normrnd(x(ii-1),sigma);  % no need to multiply by q because
%     the normal density is symmetric
    numerator = (0.3*exp(-0.2*tempX*tempX)+0.7*exp(-0.2*(tempX-10).^2));%.*...
%         normpdf(x(ii-1),tempX,sigma);
    denominator = (0.3*exp(-0.2*x(ii-1).^2)+0.7*exp(-0.2*(x(ii-1)-10).^2));%.*...
%         normpdf(tempX,x(ii-1),sigma);
    A = min(1,numerator/denominator);
    if u < A
        x(ii) = tempX;
    else
        x(ii) = x(ii-1);
    end
end

figure();histogram(x(10000:end));

% find the modes with MCMC simulated annealing
x = zeros(N,1);
x(1) = normrnd(0,sigma);
T = 1;
for ii=2:N
    u = rand;
    tempX = x(ii-1)+normrnd(0,1);
    numerator = (0.3*exp(-0.2*tempX*tempX)+0.7*exp(-0.2*(tempX-10).^2)).^(1/T);
    denominator = (0.3*exp(-0.2*x(ii-1).^2)+0.7*exp(-0.2*(x(ii-1)-10).^2)).^(1/T);
    A = min(1,numerator/denominator);
    if u < A
        x(ii) = tempX;
    else
        x(ii) = x(ii-1);
    end
    T = 1-(ii/N);
end
figure();histogram(x);mode(x)

% a gibbs sampler requires a series of conditional probability distributions
% to draw samples from