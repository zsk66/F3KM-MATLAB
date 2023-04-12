% KMeans++ clustering algorithm
% X: a N-by-D matrix of N data points in D dimensions
% K: the number of clusters to find
% centroids: a K-by-D matrix of initial centroids
% labels: a N-by-1 vector of labels indicating the cluster to which each data point belongs

function [labels, centroids] = kmeans_plusplus(X, K)

centroids = X(randperm(size(X, 1), 1), :);

% Choose the remaining centroids using KMeans++
for k = 2:K
    % Calculate the distance between each data point and the current centroids
    distances = pdist2(X, centroids);
    % Calculate the distance to the closest centroid for each data point
    min_dist = min(distances, [], 2);
    % Normalize the distances to create a probability distribution
    probs = min_dist.^2 / sum(min_dist.^2);
    % Choose the next centroid randomly, using the probability distribution
    centroids(k, :) = X(randsample(size(X, 1), 1, true, probs), :);
end

% Calculate the distance between each data point and each centroid
distances = pdist2(X, centroids);

% Assign each data point to the closest centroid
[~, labels] = min(distances, [], 2);

end

