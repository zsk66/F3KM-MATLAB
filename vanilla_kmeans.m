function [C, I, iter,RSS_error,balance_value] = vanilla_kmeans(X, K, maxIter,label,color)
% number of vectors in X
[vectors_num, dim] = size(X);
% compute a random permutation of all input vectors
R = randperm(vectors_num);
% construct indicator matrix (each entry corresponds to the cluster of each point in X)
I = zeros(vectors_num, 1);
% construct centers matrix
C = zeros(K, dim);
% take the first K points in the random permutation as the center sead
X=X';
for ii=1:K
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2); 
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
X=X';
Loss(1)= sum(sumd);    % Initial objective function value
C = center';
% iteration count
iter = 1;
% compute new clustering while the cumulative intracluster error in kept
% below the maximum allowed error, or the iterative process has not
% exceeded the maximum number of iterations permitted
while 1
    % find closest point
    for n=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = norm(X(n,:) - C(minIdx,:), 2);
        for j=1:K
            dist = norm(C(j,:) - X(n,:), 2);
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        % assign point to the closter center
        I(n) = minIdx;
    end
    % compute centers
    for k=1:K
        C(k, :) = sum(X(find(I == k), :));
        C(k, :) = C(k, :) / length(find(I == k));
    end
    % compute RSS error
    RSS_error = 0;
    for idx=1:vectors_num
        RSS_error = RSS_error + norm(X(idx, :) - C(I(idx),:), 2)^2;
    end
    % increment iteration
    iter = iter + 1;
    Loss(iter) = RSS_error;
    % check stopping criteria
    if Loss(iter)-Loss(iter-1)==0
        break;
    end
    
    if iter > maxIter
        iter = iter - 1;
        break;
    end
end
RSS_error = sqrt(RSS_error);
F = sparse(1:n,I,1,n,K,n); 
P=[];
top_k = 4;
for i = 1:size(color,2)
    I = eye(max(color(:,i)'+1));
    P_tmp = I(color(:,i)'+1,:);
    P=[P,P_tmp];
end
 balance_value = compute_balance(F,P,top_k);
disp(['k-means took ' int2str(iter) ' steps to converge']);