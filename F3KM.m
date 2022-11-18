
function [Y, minO, iter_num, obj,balance_value] = F3KM(X, label,c, color, delta,block_size)
% Input
% X d*n data
% color the color vector
% delta is delta in our paper
% block_size the block_size in our paper
% label is initial label n*1
% c is the number of clusters
% Output
% Y is the label vector n*1
% minO is the Converged objective function value
% iter_num is the number of iteration
% obj is the objective function value
% balance_value is balance in our paper
P=[];alpha=[];beta=[];
[~,n] = size(X);
block_num = ceil(n/block_size);   
F = sparse(1:n,label,1,n,c,n); 
top_k = 4;
violation = 3;
iter_num = 0;
for i = 1:size(color,2)
    I = eye(max(color(:,i)'+1));
    P_tmp = I(color(:,i)'+1,:);
    P=[P,P_tmp];
end
P = sparse(P);
[~,l] = size(P);
for i = 1:size(P,2)
    alpha(i) = sum(P(:,i))/ (n*(1-delta));
    beta(i) = sum(P(:,i))*(1-delta)/ n;
end


%% compute Initial objective function value
for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2); 
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
obj(1)= sqrt(sum(sumd));    % Initial objective function value
%% store once
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end    
XF = X*F;
FF=sum(F,1);    % diag(F'*F) ;
FXXF=XF'*XF;    % F'*X'*X*F;
PF = P'*F;      % PF in our paper
for j = 1:c
    for i = 1:l
        theta(i,j) = (P(:,i)' * F(:,j))/FF(:,j);
    end
end

theta = projection(theta,alpha,beta);  
u =5e2 * rand(l,c);  % Lagrange multipliers  
rho =1e4* ones(l,c);   % Penalty factor    
stop = 1;
iter=0;
while stop
% while any(label ~= last) 
    iter = iter +1;
    last = label;
    phi=[];
    UnFairPhi=[];
    m = [];
    lagrange_sum=zeros(n,c);
    augmented_lagrange_sum=zeros(n,c);
%     rho =(5e3*(iter^0.5)) * ones(l,c);  % varied step size
    
%% Solve F
    for i = 1:n
        delta = sparse(n,1); 
        delta(i,1) = 1;
        m = label;   
        for k = 1:c     
            if k == m(i,:)   
                V1(i,k) = FXXF(k,k)- 2 * X(:,i)'* XF(:,k);
                PF_div_F1 = PF(:,k)/FF(k);
                PF_div_F2 = (PF(:,k)-P'* delta) / (FF(k)-1);
                PF_sub = PF_div_F1-PF_div_F2;
                PF_sum = PF_div_F1+PF_div_F2;
                lagrange_sum(i,k) = -u(:,k)'*PF_sub;
                augmented_lagrange_sum(i,k) = (rho(:,k)'/2)*(PF_sub.*(PF_sum-2*theta(:,k)));
                U1(i,k) = V1(i,k) / (FF(k) -1) - FXXF(k,k) / FF(k);
                phi(i,k) = U1(i,k) + lagrange_sum(i,k) + augmented_lagrange_sum(i,k); 
            else  
                V2(i,k) =(FXXF(k,k)  + 2 * X(:,i)'* XF(:,k));
                PF_div_F1 = PF(:,k)/FF(k);
                PF_div_F2 = (PF(:,k)+P'* delta) / (FF(k)+1);
                PF_sub = PF_div_F2 - PF_div_F1;
                PF_sum = PF_div_F2 + PF_div_F1;
                lagrange_sum(i,k) = -u(:,k)'*PF_sub;
                augmented_lagrange_sum(i,k) = (rho(:,k)'/2)*(PF_sub.*(PF_sum-2*theta(:,k)));
                U2(i,k) = FXXF(k,k)/ FF(k) -  V2(i,k) / (FF(k) +1);
                phi(i,k) = U2(i,k) + lagrange_sum(i,k) + augmented_lagrange_sum(i,k); 
            end 
        end
        
        if (rem(i/block_size,1)==0)||(i==n)
            [~,label_update] = min(phi,[],2);
            q = find(m(1:i)~=label_update)';
            for j = q
                 XF(:,label_update(j))=XF(:,label_update(j))+X(:,j); 
                 XF(:,m(j))=XF(:,m(j))-X(:,j); 
                 FF(label_update(j))= FF(label_update(j)) +1; 
                 FF(m(j))= FF(m(j)) -1;
            end  
            label(1:i,:)=label_update;
            FXXF=XF'*XF; 
            F = sparse(1:n,label,1,n,c,n);
            PF = P'*F;
        end       
    end
%% Solve theta
    for i = 1:l
        for j = 1:c
            theta(i,j)=PF(i,j) / FF(j) - u(i,j) / rho(i,j);
        end
    end
    theta = projection(theta,alpha,beta); 
%% Solve u
    for i = 1:l
        for j = 1:c
            u(i,j) = u(i,j)+rho(i,j)*(theta(i,j)-PF(i,j)/FF(j));
        end
    end
    iter_num = iter_num+1;
%% compute objective function value
    for ii=1:c
        idxi = label==ii;
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii,1) = sum(d2c); 
    end
    obj(iter_num+1) = sqrt(sum(sumd)) ;     
    fprintf('obj=%f\n',obj(iter_num+1))
    stop1 = stopping_criteria(F,P,alpha,beta,violation);
    stop2=0;
    if obj(iter_num+1)<77
        stop2=1;
    end
    stop = ~(stop1&&stop2);
end
balance_value = compute_balance(F,P,top_k);
minO=obj(iter_num+1);
Y=label;
end
