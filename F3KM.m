
function [Y, minO, iter_num, obj,balance_value] = F3KM(X, label,c, color, delta,block_size,rho_0,u_0,violation,max_iters)
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
pool = parpool("local",6);
P=[];alpha=[];beta=[];
[~,n] = size(X);
block_num = ceil(n/block_size);   
F = sparse(1:n,label,1,n,c,n); 
top_k = c;
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

theta = rand(l,c);
theta = projection(theta,alpha,beta);  
u =u_0 * rand(l,c);  % Lagrange multipliers  
rho =rho_0* ones(l,c);   % Penalty factor   
PF = P'*F;      % PF in our paper
PFdivFF = PF./ repmat(FF,size(PF,1),1);
PFdivFFplus1 = PF./ repmat(FF+1,size(PF,1),1);
PFdivFFsub1 = PF./ repmat(FF-1,size(PF,1),1);

PFdivFFsub1theta = rho .* (PFdivFFsub1-theta);
PFdivFFplus1theta = rho .* (PFdivFFplus1-theta);

thetasub = PFdivFF + PFdivFFsub1 - 2*theta;
thetaplus = PFdivFF + PFdivFFplus1 - 2*theta;

sub = (PFdivFFsub1 - PFdivFF);
plus = (PFdivFF - PFdivFFplus1);

lag_sub = sum(u.* sub,1)+sum((rho/2).* ((-sub).* thetasub),1);
lag_plus = sum(u.* plus,1)+sum((rho/2).* ((-plus).* thetaplus),1);

stop = 0;
iter=0;
blocks = partitionNumbers(n,block_size);
while stop==0

    iter = iter +1;
    phi=zeros(1,c);
    rho =(rho_0*(iter^0.5)) * ones(l,c);  % varied step size
%% Solve F
    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;   
        parfor idx = 1:length(block)
            i = block(idx);
            for k = 1:c     
                if k == m(i,:)   
                    V1 = FXXF(k,k)- 2 * X(:,i)'* XF(:,k)+ XX(i);
                    U1 = V1/ (FF(k) -1) - FXXF(k,k) / FF(k);
                    S1 = P(i,:) / (FF(k)-1);
                    W1 = lag_sub(k) + S1 * PFdivFFsub1theta(:,k) - S1 * u(:,k)- S1.*S1 * rho(:,k)/2;
                    phi1(idx,k) = U1 + W1; 
                else  
                    V2 =(FXXF(k,k)  + 2 * X(:,i)'* XF(:,k)+ XX(i));
                    U2 = FXXF(k,k)/ FF(k) -  V2 / (FF(k) +1);
                    S2 = P(i,:) / (FF(k)+1);
                    W2 = lag_plus(k) + S2 * PFdivFFplus1theta(:,k) - S2 * u(:,k) + S2.*S2*rho(:,k)/2;
                    phi1(idx,k) = U2 + W2; 
                end 
            end
        end
        phi(block,:) = phi1(1:length(block),:);
        [~,label_update] = min(phi,[],2);
        q = find(m(1:block(end))~=label_update)';
        for j = q
             XF(:,label_update(j))=XF(:,label_update(j))+X(:,j); 
             XF(:,m(j))=XF(:,m(j))-X(:,j); 
             FF(label_update(j))= FF(label_update(j)) +1; 
             FF(m(j))= FF(m(j)) -1;
             PF(:,label_update(j))=PF(:,label_update(j))+P(j,:)';
             PF(:,m(j))=PF(:,m(j))-P(j,:)';
        end  
        label(1:block(end),:)=label_update;
        FXXF=XF'*XF; 
        F = sparse(1:n,label,1,n,c,n);
        PFdivFF = PF./ repmat(FF,size(PF,1),1);
        PFdivFFplus1 = PF./ repmat(FF+1,size(PF,1),1);
        PFdivFFsub1 = PF./ repmat(FF-1,size(PF,1),1);
        PFdivFFsub1theta = rho .* (PFdivFFsub1-theta);
        PFdivFFplus1theta = rho .* (PFdivFFplus1-theta);
        thetasub = PFdivFF + PFdivFFsub1 - 2*theta;
        thetaplus = PFdivFF + PFdivFFplus1 - 2*theta;
        sub = (PFdivFFsub1 - PFdivFF);
        plus = (PFdivFF - PFdivFFplus1);
        lag_sub = sum(u.* sub,1)+sum((rho/2).* ((-sub).* thetasub),1);
        lag_plus = sum(u.* plus,1)+sum((rho/2).* ((-plus).* thetaplus),1);

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
    PFdivFFsub1theta = rho .* (PFdivFFsub1-theta);
    PFdivFFplus1theta = rho .* (PFdivFFplus1-theta);
    thetasub = PFdivFF + PFdivFFsub1 - 2*theta;
    thetaplus = PFdivFF + PFdivFFplus1 - 2*theta;
    sub = (PFdivFFsub1 - PFdivFF);
    plus = (PFdivFF - PFdivFFplus1);
    lag_sub = sum(u.* sub,1)+sum((rho/2).* ((-sub).* thetasub),1);
    lag_plus = sum(u.* plus,1)+sum((rho/2).* ((-plus).* thetaplus),1);
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
    fprintf('obj=%f\n',obj(iter_num+1)^2)
    if iter_num>max_iters
        stop1 = 1;
    else stop1 = 0;
    end
    balance_value = compute_balance(F,P,top_k);
    stop2 = stopping_criteria(F,P,alpha,beta,violation);
    stop = stop1 || stop2;
%     lagrange_loss = sum(phi,'all');
%     fprintf('lag_loss=%f\n',lagrange_loss)

end

minO=obj(iter_num+1)^2;
Y=label;
balance_value = full(balance_value);
delete(gcp('nocreate'))
end
