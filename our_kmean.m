test = [[1,2,3];
        [2,2,2];
        [2,3,1];
        [4,2,3];
        [1,1,1];
        [2,3,3]];

result = our_kmeans(test, 2, [[1,1,1];[3,3,3]], 50);

function result = our_kmeans(data, num_clusters, start_centres, max_iter)
    m = size(data,1); % number of data points
    n = size(data,2); % number of features
    
    result = zeros(m,1); % each data point will be assigned a cluster
    distance = zeros(m,num_clusters); % this will store the distance to each cluster center
    centres = start_centres;

    % while loop
    for i = 1:num_clusters
        distance(:,i) = sum(abs(data - repmat(centres(i),m,n)),2);
    end
    [~,I] = min(distance,[],2);
    if I == result
        break
    else
        centres = 

        for i = 1:m
            centres
        end
    end
  
end