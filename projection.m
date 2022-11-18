function theta = projection(theta,alpha,beta)
        for i = 1:length(alpha)
            theta(i,:) = min(alpha(i),max(theta(i,:),beta(i)));
        end
end