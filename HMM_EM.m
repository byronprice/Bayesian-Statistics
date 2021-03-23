function [logProbPath,states,Pi,P,EmissionDist] = HMM_EM(data,K)
% HMM_EM.m
%   EM algorithm to fit the parameters of a discrete-state Hidden Markov
%    Model
%
%  certain scenarios warrant a so-called left-to-right transition
%   probability matrix, in that case, simply set the matrix initially to be
%   an upper triangular matrix, if the probabilities start at zero, they
%   will remain at 0

[N,d] = size(data);

logData = log(data);

% start with Gaussian Mixture Model to initialize emission distribution and
%  transition probability matrix
[Pi,mu,sigma,logalpha] = GaussMixtureEM(data,K);

states = zeros(N,1);

for nn=1:N
    [~,ind] = max(logalpha(nn,:));
    states(nn) = ind;
end

EmissionDist = cell(K,2);
for kk=1:K
    EmissionDist{kk,1} = real(mu{kk});
    EmissionDist{kk,2} = real(sigma{kk});
end

P = zeros(K,K);

for nn=2:N
    currentState = states(nn);
    prevState = states(nn-1);
    P(prevState,currentState) = P(prevState,currentState)+1;
end

for kk=1:K
    P(kk,:) = P(kk,:)./sum(P(kk,:));
end

maxIter = 1e4;
tolerance = 1e-6;
prevLikelihood = -Inf;
for iter=1:maxIter
    % E step
    %   calculate responsibilities and soft transition matrix
    [currentLikelihood,logalpha] = ForwardHMM(P,EmissionDist,Pi,data);
    [logbeta] = BackwardHMM(P,EmissionDist,data);
    
    logepsilon = zeros(N-1,K,K);
    for nn=2:N
        for jj=1:K
            currentAlpha = logalpha(nn-1,jj);
            for kk=1:K
                logxgivenz = LogMvnPDF(data(nn,:)',...
                    EmissionDist{kk,1},EmissionDist{kk,2});
                logzgivenz = log(P(jj,kk));
                logepsilon(nn-1,jj,kk) = currentAlpha+logxgivenz+...
                    logzgivenz+logbeta(nn,kk);
            end
        end
    end
    
    % M step
    %    update model parameters
    
    % pi update, probability of first state to start the chain
    normalization = LogSum(logalpha(1,:)+logbeta(1,:),K);
    for kk=1:K
        Pi(kk) = exp(logalpha(1,kk)+logbeta(1,kk)-normalization);
    end
    
    % P update, transition probability matrix
    for jj=1:K
        tmp = squeeze(logepsilon(:,jj,:));
        normalization = LogSum(tmp(:),(N-1)*K);
        for kk=1:K
            P(jj,kk) = exp(LogSum(tmp(:,kk),N-1)-normalization);
        end
    end
    
    % emissions update, state emission distribution
    for kk=1:K
        for dd=1:d
           mu{kk}(dd) = exp(LogSum(logalpha(:,kk)+logbeta(:,kk)+logData(:,dd),N)...
               -LogSum(logalpha(:,kk)+logbeta(:,kk),N)); 
        end
        
        tmp = zeros(N,d,d);
        for nn=1:N
            tmp(nn,:,:) = log((data(nn,:)'-mu{kk})*(data(nn,:)'-mu{kk})');
        end
        
        for jj=1:d
            for dd=1:d
                sigma{kk}(jj,dd) = exp(LogSum(logalpha(:,kk)+logbeta(:,kk)+...
                    squeeze(tmp(:,jj,dd)),N)-LogSum(logalpha(:,kk)+logbeta(:,kk),N));
            end
        end
        
        EmissionDist{kk,1} = real(mu{kk});
        EmissionDist{kk,2} = real(sigma{kk});
    end

%     [currentLikelihood,~] = ForwardHMM(P,EmissionDist,Pi,data);
    
    if currentLikelihood-prevLikelihood > tolerance
        prevLikelihood = currentLikelihood;
    else
        break;
    end
end

% calculate most probable sequence of states
[logProbPath,states] = ViterbiHMM(P,EmissionDist,Pi,data);

% calculate steady state probabilities
[~,D,V] = eig(P);
D = diag(D);
[~,index] = min(abs(D-1));
steadyState = V(:,index)./sum(V(:,index));

end

function [logPDF] = LogMvnPDF(data,mu,sigma)
logdet = 2*sum(log(diag(chol(sigma))));
logPDF = -0.5*logdet-0.5*(data-mu)'*(sigma\(data-mu));

end

function [summation] = LogSum(vector,vectorLen)
if vectorLen==0
    summation = -Inf;
elseif vectorLen==1
    summation = vector(1);
else
    vector = sort(vector);
    summation = LogSumExpTwo(vector(1),vector(2));
    for ii=2:vectorLen-1
        summation = LogSumExpTwo(summation,vector(ii+1));
    end
end

end

function [y] = LogSumExpTwo(x1,x2)
check = x1>=x2;
if check==1
    y = x1+SoftPlus(x2-x1);
else
    y = x2+SoftPlus(x1-x2);
end
end

function [y] = SoftPlus(x)
if x<-34 % condition for small x
   y = 0;
else
   y = log(1+exp(-x))+x; % numerically stable calculation of log(1+exp(x))
end

end