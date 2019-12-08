% VisualizeTDR.m

[Z,X,hk,Wtrue,Strue,Dtrue,ranktrue] = SimulateTDRData;

[W,S,b,D,rank] = TDR_Gibbs(Z,X,hk);

[N,T,~] = size(Z);
% calculate proportion of variance explained
varExp = zeros(N,1);
for nn=1:N
   data = squeeze(Z(nn,:,logical(hk(:,nn))));
   varExp(nn) = 1-mean(D(nn,:),2)/var(data(:));
end

P = length(rank);

order = cell(P,1);
order{1} = [1];
order{2} = [1];
order{3} = [2,1];
order{4} = [3,1,2];
for pp=1:P
    figure;
    for jj=1:rank(pp)
        subplot(2,rank(pp),jj);plot(Strue{pp}(:,jj));axis([0 T -4 4]);
        title(sprintf('True Basis, p:%d',pp));
        xlabel('Time');
        for kk=randperm(1000,250)
            tmp = S{kk};
            if P==4 && jj==1
                subplot(2,rank(pp),jj+rank(pp));plot(-tmp{pp}(:,order{pp}(jj)));hold on;
            elseif P==4 && jj==2
                subplot(2,rank(pp),jj+rank(pp));plot(-tmp{pp}(:,order{pp}(jj)));hold on;
            else
                subplot(2,rank(pp),jj+rank(pp));plot(tmp{pp}(:,order{pp}(jj)));hold on;
            end
            title(sprintf('Inferred Basis, p:%d',pp));
            xlabel('Time');
            axis([0 T -4 4]);
        end
    end
end

% example neuron
nn = randperm(N,1);
tmp = squeeze(Z(nn,:,logical(hk(:,nn))));

figure;subplot(2,1,1);imagesc(tmp');xlabel('Time');ylabel('Trial');
title('Example Data');colormap('jet');
subplot(2,1,2);plot(mean(tmp,2));xlabel('Time');ylabel('FR (AU)');
title('Average Response');

hold on;

meanResp = zeros(T,1);

for ii=1:1000
    meanResp = meanResp+b(nn,ii)+S{ii}{4}*W{ii}{4}(nn,:)';
end

meanResp = meanResp./1000;
plot(meanResp);

legend('Data','Model');
    