function [Y, minO, iter_num, obj,balance_value] = BCDKM(X, label,c,block_size,color)
[~,n] = size(X);
block_num = ceil(n/block_size);
F = sparse(1:n,label,1,n,c,n);  
last = 0;
iter_num = 0;
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
FF = sum(F,1);% diag(F'*F) ;
FXXF = XF'*XF;% F'*X'*X*F;
%% compute F
while any(label ~= last)   
    last = label;
    delta = [];
    m = [];
 for i = 1:n
    m = label;
    for k = 1:c        
        if k == m(i)   
           V1(i,k) = - 2 * X(:,i)'* XF(:,k);
           delta(i,k) = FXXF(k,k) / FF(k) - (FXXF(k,k)+V1(i,k)) / (FF(k) -1); 
        else  
           V2(i,k) =2 * X(:,i)'* XF(:,k);
           delta(i,k) =(FXXF(k,k)+V2(i,k)) / (FF(k) +1) -  FXXF(k,k)  / FF(k); 
        end         
    end
    if (rem(i/block_size,1)==0)||(i==n)
        [~,label_update] = max(delta,[],2);
        q = find(m(1:i)~=label_update)';
        for j = q
             XF(:,label_update(j))=XF(:,label_update(j))+X(:,j); 
             XF(:,m(j))=XF(:,m(j))-X(:,j); 
             FF(label_update(j))= FF(label_update(j)) +1; 
             FF(m(j))= FF(m(j)) -1; 
        end  
        label(1:i,:)=label_update;
        FXXF=XF'*XF;% F'*X'*X*F;   
    end
 end   
  iter_num = iter_num+1;
%% compute objective function value
   for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii,1) = sum(d2c); 
    end
    obj(iter_num+1) = sqrt(sum(sumd)) ;     %  objective function value     
end    
 minO=min(obj);
 Y=label;
 P=[];
 F = sparse(1:n,label,1,n,c,n); 
 top_k = 4;
for i = 1:size(color,2)
    I = eye(max(color(:,i)'+1));
    P_tmp = I(color(:,i)'+1,:);
    P=[P,P_tmp];
end
 balance_value = compute_balance(F,P,top_k);
end