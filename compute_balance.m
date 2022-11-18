function balance_value = compute_balance(F,P,top_k)
    [n,~]=size(F);
    sum_P=sum(P,1);
    sum_F=sum(F);
    [~,clusters_sorted]=sort(sum(F),'descend');
    top_k_clusters = clusters_sorted(:,1:top_k);
    r_i = sum_P/n;
    balance_value=[];
    for k = top_k_clusters
        idx = F(:,k)==1;
        color_in_cluster_k = P(idx,:);
        C_i_f = sum(color_in_cluster_k);
        C_f = sum_F(:,k);
        r_i_f = C_i_f / C_f;
        balance_f = min(min(r_i_f./r_i,r_i_f.\r_i));
        balance_value = [balance_value,balance_f];
    end
end