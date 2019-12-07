% VisualizeTDR.m

[Z,X,hk,Wtrue,Strue,Dtrue,ranktrue] = SimulateTDRData;

[W,S,b,D,rank] = TDR_Gibbs(Z,X,hk);

P = length(rank);

order = cell(P,1);
order{1} = [1];
order{2} = [1];
order{3} = [1];
order{4} = [1];
for pp=1:P
    figure;
    for jj=1:rank(ii)
        subplot(2,rank(ii),jj);plot(Strue{ii}(:,jj));
        for kk=1:1000
            tmp = S{kk};
            subplot(2,rank(ii),jj+rank(ii));plot(tmp{ii}(:,jj));hold on;
        end
    end
end

