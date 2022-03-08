function result = our_kmeans(data, num_clusters, start_centres, max_iter)
    m = size(data,1); % number of data points
    n = size(data,2); % number of features
    
    result   = zeros(m,1); % each data point will be assigned a cluster
    distance = zeros(m,num_clusters); % this will store the distance to each cluster center
    centres  = start_centres;

    iter = 0;
    while iter < max_iter % Stop if iterations surpass max
        for i = 1:num_clusters
            % Calculate distance to each cluster
            distance(:,i) = sum(abs(data - repmat(centres(i,:),m,1)),2);
        end
        
        % Find cluster at minimum distance to each point
        [~,I] = min(distance,[],2);

        if I == result % Stop if clusters don't change
            break
        end
        
        % Calculate new centers: find center of points
        centres = zeros(num_clusters,n);
        counter = zeros(num_clusters,1);
        for i = 1:m
            cluster = I(m);
            centres(cluster,:) = centres(cluster,:) + data(m,:);
            counter(cluster) = counter(cluster) + 1;
        end
        
        for i = 1:num_clusters
            if counter(i) > 0
                centres(i,:) = centres(i,:) / counter(i);
            end
        end

        result = I;
        iter = iter + 1;
    end
  
end

