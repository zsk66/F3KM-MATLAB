clc
clear
dataset_name = 'bank';
if strcmp(dataset_name,'bank')
    X = csvread('F3KM/F3KM-MATLAB/bank/bank.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/bank/bank_color.csv',1,1);
elseif strcmp(dataset_name,'adult')
    X = csvread('F3KM/F3KM-MATLAB/adult/adult.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/adult/adult_color.csv',1,1);
elseif strcmp(dataset_name,'creditcard')
    X = csvread('F3KM/F3KM-MATLAB/creditcard/creditcard.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/creditcard/creditcard_color.csv',1,1);
elseif strcmp(dataset_name,'census1990')
    X = csvread('F3KM/F3KM-MATLAB/census1990/census1990.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/census1990/census1990_color.csv',1,1); 
elseif strcmp(dataset_name,'diabetes')
    X = csvread('F3KM/F3KM-MATLAB/diabetes/diabetes.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/diabetes/diabetes_color.csv',1,1); 
elseif strcmp(dataset_name,'athlete')
    X = csvread('F3KM/F3KM-MATLAB/athlete/athlete.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/athlete/athlete_color.csv',1,1); 
elseif strcmp(dataset_name,'recruitment')
    X = csvread('F3KM/F3KM-MATLAB/recruitment/recruitment.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/recruitment/recruitment_color.csv',1,1); 
elseif strcmp(dataset_name,'Spanish')
    X = csvread('F3KM/F3KM-MATLAB/Spanish/Spanish.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/Spanish/Spanish_color.csv',1,1); 
elseif strcmp(dataset_name,'student')
    X = csvread('F3KM/F3KM-MATLAB/student/student.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/student/student_color.csv',1,1); 
elseif strcmp(dataset_name,'hmda')
    X = csvread('F3KM/F3KM-MATLAB/hmda/hmda.csv',1,1);
    color = csvread('F3KM/F3KM-MATLAB/hmda/hmda_color.csv',1,1); 
end

[n, d] = size(X);
block_size =1000;
c = 4;
X=X';
max_iters = 100;
delta = 0.2;
rho_0 =1e3; 
u_0 = 5e2;  
violation = 1;
% label_rnd = (randsrc(n,1,1:c));    % Random initialization
label_kmeans_plusplus = kmeans_plusplus(X', c);

num_iters = 1;
top_k = c;


% [Y, minO, iter_num, obj,balance_value] = CDKM(X, label_kmeans_plusplus,c,color,top_k);

[F_fair, label_fair, iter_num_fair, f0_fair,balance_value_fair] = F3KM(X,label_kmeans_plusplus, c, color,delta,block_size,rho_0,u_0,violation,max_iters);  % F3KM


