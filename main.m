clc
clear
dataset_name = 'bank';
if strcmp(dataset_name,'bank')
    X = csvread('/bank/bank.csv',1,1);
    color = csvread('/bank/bank_color.csv',1,1);
elseif strcmp(dataset_name,'adult')
    X = csvread('/adult/adult.csv',1,1);
    color = csvread('/adult/adult_color.csv',1,1);
elseif strcmp(dataset_name,'creditcard')
    X = csvread('/creditcard/creditcard.csv',1,1);
    color = csvread('/creditcard/creditcard_color.csv',1,1);
elseif strcmp(dataset_name,'census1990')
    X = csvread('/census1990/census1990.csv',1,1);
    color = csvread('/census1990/census1990_color.csv',1,1); 
elseif strcmp(dataset_name,'diabetes')
    X = csvread('/diabetes/diabetes.csv',1,1);
    color = csvread('/diabetes/diabetes_color.csv',1,1); 
end

% data = load('dataset/data 1/Mpeg7_uni');  
% X = data.X;
% X = mapminmax(X,0,1);          % normalize X to [0,1]

c = 4;
delta = 0.2;
[n, d] = size(X);
block_size =1000;
X=X';
maxIter = 50;
label_rnd = randsrc(n,1,1:c);    % Random initialization
[C, I, iter,obj,balance_value_Lloyd] = vanilla_kmeans(X', c, maxIter,label_rnd,color);   % Lloyd heuristic
[F_bcd, label_bcd, iter_num_bcd, f0_bcd,balance_value_bcd] =BCDKM(X, label_rnd, c,block_size,color);  % block coordinate descent for k-means
[F_fair, label_fair, iter_num_fair, f0_fair,balance_value_fair] =F3KM(X,label_rnd, c, color,delta,block_size);  % F3KM
