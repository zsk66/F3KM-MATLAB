function theta = projection(theta,alpha,beta)
[L,K] = size(theta);
    for l = 1:L
        for j = 1:K
            if theta(l,j)>alpha(l)
                theta(l,j) = alpha(l);
            else if theta(l,j)<beta(l)
                theta(l,j) = beta(l);
            else continue
            end
                    
        end
    end
end