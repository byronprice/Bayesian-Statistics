function [finalParameters,covariance] = BayesianFitLFPModel(Response,xaxis,yaxis,centerVals)
% BayesianFitLFPModel.m
tic
numChans = size(Response,1);
reps = size(Response,2);

numParameters = 7;
   
finalParameters = zeros(numChans,numParameters);
covariance = zeros(numChans,numParameters,numParameters);

priorParams = [1.75,100;4,300;4,250;1.75,200;1.75,200;1.5,200;1.75,200];
% proposal = [150,100;1000,500;700,500;300,200;250,300;200,150;200,150];

initSigma = [1,2,2,1,1,1,1].*5;

for zz=1:numChans
    flashPoints = centerVals(squeeze(Response(zz,:,1)),:);
    peakNegativity = squeeze(Response(zz,:,2));
    N = 2e6; burnIn = 1e6;
    x = zeros(N,numParameters);
    rejectRate = zeros(N,1);
%     [finalParams,finalLikelihood] = GradientAscentInit(flashPoints,peakNegativity,reps);
%      x(1,:) = finalParams;
    for jj=1:numParameters
        x(1,jj) = gamrnd(priorParams(jj,1),priorParams(jj,2));
    end
    
    
    likelihoodXprev = GetLikelihood(reps,x(1,:),peakNegativity,flashPoints);
    
    sigma = initSigma;
    for ii=2:N
%         priorXstar = 1;priorXprev = 1;
        xStar = x(ii-1,:)+normrnd(0,sigma,[1,numParameters]);

%         for jj=1:numParameters
% %             priorXstar = priorXstar*gampdf(xStar(jj),priorParams(jj,1),priorParams(jj,2));
% %             priorXprev = priorXprev*gampdf(x(ii-1,jj),priorParams(jj,1),priorParams(jj,2));
%         end
        if sum(xStar<0) > 0
            x(ii,:) = x(ii-1,:);
            rejectRate(ii) = 1;
        else
            likelihoodXstar = GetLikelihood(reps,xStar,peakNegativity,flashPoints);
            %         A = min(1,(likelihoodXstar*priorXstar)./(likelihoodXprev*priorXprev));
%             A = min(1,likelihoodXprev/likelihoodXstar);
            logA = likelihoodXstar-likelihoodXprev;
            if log(rand) < logA
                x(ii,:) = xStar;
                likelihoodXprev = likelihoodXstar;
            else
                x(ii,:) = x(ii-1,:);
                rejectRate(ii) = 1;
            end
        end
    end
    sum(rejectRate(burnIn:end))./(N-burnIn)
    figure(1);figure(2);
    for ii=1:numParameters
       figure(1);subplot(4,2,ii);histogram(x(burnIn:500:end,ii));
       figure(2);subplot(4,2,ii);autocorr(x(burnIn:500:end,ii));
    end
    finalParameters(zz,:) = mean(x(burnIn:500:end,:),1);
    median(x(burnIn:end,:))
    covariance(zz,:,:) = cov(x(burnIn:500:end,:));
end

toc
end

function [loglikelihood] = GetLikelihood(reps,parameterVec,peakNegativity,flashPoints)
loglikelihood = 0;
for kk=1:reps
      mu = parameterVec(1)*exp(-((flashPoints(kk,1)-parameterVec(2)).^2)./(2*parameterVec(4).^2)-...
        ((flashPoints(kk,2)-parameterVec(3)).^2)./(2*parameterVec(5).^2))+parameterVec(7);

    loglikelihood = loglikelihood-(1/2)*log(2*pi*parameterVec(6)*parameterVec(6))-...
        (1/(2*parameterVec(6)*parameterVec(6)))*(peakNegativity(kk)-mu).^2;
end
end

function [finalParams,finalLikelihood] = GradientAscentInit(flashPoints,peakNegativity,reps)
%GradientAscentInit

numParameters = 7;
maxITER = 1000;
tolerance = 1e-2;

h = ones(numParameters,1)./100;

% repeat gradient ascent from a number of different starting
%  positions
gradientVec = zeros(1,numParameters);
parameterVec = zeros(maxITER,numParameters);
logLikelihood = zeros(maxITER,1);

priorParams = [1.75,100;4,300;4,250;1.75,200;1.75,200;1.5,200;1.75,200];
for jj=1:numParameters
    parameterVec(1,jj) = gamrnd(priorParams(jj,1),priorParams(jj,2));
end


logLikelihood(1) = GetLikelihood(reps,parameterVec(1,:),peakNegativity,flashPoints);
check = 1;
iter = 1;

while check > tolerance && iter < maxITER
    iter = iter+1;
    % calculate likelihood at the current position in parameter
    %  space
    
    % calculate the gradient by calculating the likelihood
    %  after moving over a small step h along each
    %  dimension
    for jj=1:numParameters
        tempParameterVec = parameterVec(iter-1,:);
        tempParameterVec(jj) = tempParameterVec(jj)+h(jj);
        [gradLikelihoodplus] = GetLikelihood(reps,tempParameterVec,peakNegativity,flashPoints);
        
        tempParameterVec = parameterVec(iter-1,:);
        tempParameterVec(jj) = tempParameterVec(jj)-h(jj);
        [gradLikelihoodminus] = GetLikelihood(reps,tempParameterVec,peakNegativity,flashPoints);
        gradientVec(jj) = (gradLikelihoodplus-gradLikelihoodminus)./(2*h(jj));
    end
    
    % line search to get distance to move along gradient
    alpha = [0,1e-6,1e-4,1e-2,1e-1];
    lineSearchLikelihoods = zeros(length(alpha),1);
    lineSearchLikelihoods(1) = logLikelihood(iter-1);
    
    for ii=2:length(alpha)
        tempParameterVec = parameterVec(iter-1,:)'+gradientVec.*alpha(ii);
        [lineSearchLikelihoods(ii)] = GetLikelihood(reps,tempParameterVec,peakNegativity,flashPoints);
    end
    
    newInds = find(imag(lineSearchLikelihoods)==0);
    newValues = zeros(length(newInds),1);
    for ii=1:length(newInds)
        newValues(ii) = lineSearchLikelihoods(newInds(ii));
    end
    [~,ind] = max(newValues);
    ind = newInds(ind);
    
    logLikelihood(iter) = lineSearchLikelihoods(ind);
    check = diff(logLikelihood(iter-1:iter));
    parameterVec(iter,:) = parameterVec(iter-1,:)+alpha(ind).*gradientVec;
end

finalParams = parameterVec(iter,:);
finalLikelihood = logLikelihood(iter);
end