# F3KM-MATLAB
MATLAB code for paper "F3KM: Federated, Fair and fast k-means"
## Introduction

This repo holds the source code and scripts for reproducing the key experiments of our paper: F3KM: Federated, Fair, and Fast k-means.

## Datasets

Download the following datasets, and run our `data_process.py`, you can get the data format (DATASET_NAME.csv, DATASET_NAME_color.csv) that can be used in our Matlab codes. Then put them in DATASET_NAME folder. For instance, for Bank, one should put them in Bank/bank.csv and Bank/bank_color.csv.

Athlete        https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results  

Bank           https://archive.ics.uci.edu/ml/datasets/bank+marketing 

Census         https://archive.ics.uci.edu/ml/datasets/Adult

Creditcard     https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper.pdf

Diabetes       https://archive.ics.uci.edu/ml/datasets/diabetes

Recruitment    https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/

Spanish        https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/

Student        https://analyse.kmi.open.ac.uk/open_dataset

Census1990     https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper

HMDA           https://ffiec.cfpb.gov/data-browser/

## Start

There are 2 functions/APIs to be used from the files here (F3KM and CDKM). Try running `main.m`. See main.m for an example on how to use all 2 function. 

## Function 1: F3KM
Inputs: X, label,c, color, delta, block_size ,rho_0, u_0, violation, max_iters
Description of Inputs:
  X: dataset.  
  
  label: initial label of each point, we use kmeans++ for initialization.
  
  c: number of clusters.
  
  color: the color matrix P in our paper.
  
  delta: a hyperparameter for tuning alpha and delta.
  
  block_size: n_b in our paper, the number of points in each block.
  
  rho_0: the step size.
  
  u_0: the initial value of dual variable.
  
  violation: the additive violation in our paper.
  
  max_iters: the maximun iterations.
  
Return:
The function returns five numeric values. The first is the final label vector, the second is the objective function value, the third is the number of iterations, the fourth is the objective function values in each iteration, the fifth is the balance values in each cluster.

## Function 2: CDKM
Inputs: X, label,c, color

For description of inputs, view above. See sample usage in main.m

Return:
Y, minO, iter_num, obj, balance_value



