function stop = stopping_criteria(F,P,alpha,beta,violation)
    [n,c]=size(F);
    [~,l]=size(P);
    sum_F=sum(F);
    [~,clusters_sorted]=sort(sum(F),'descend');
    ratio=[];
    TorF=[];
    sum_colors = zeros(1,l);
    for k = clusters_sorted
        FairSet=[];
        idx = F(:,k)==1;
        color_in_cluster_k = P(idx,:);
        sum_colors = sum(color_in_cluster_k,1);
        Cf = sum_F(:,k);
        ratio_k = sum_colors / sum_F(:,k);
        for i = 1:l
            if (beta(i)*Cf-violation<=sum_colors(:,i))&&(sum_colors(:,i)<=alpha(i)*Cf+violation)&&(sum_colors(:,i)>0)
                Fair = 1;
            else
                Fair = 0;
            end
            FairSet=[FairSet,Fair];
        end
        if sum(FairSet)==l
            TorF = [TorF,1];
        else
            TorF = [TorF,0];
        end
    stop = min(TorF);
end