function [summation] = LogSum(vector,vectorLen)
% calculate log of sum of exponentials
%   ie log(sum[ exp(vector) ] )
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

% maxVal = max(vector);
% difference = vector-maxVal;
% summation2 = maxVal+log1p(sum(exp(difference))-1);
% 
% abs(summation2-summation)
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