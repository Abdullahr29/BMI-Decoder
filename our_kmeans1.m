function [result, centres] = our_kmeans(data, K, start_centres, max_iter)
    M = size(data,1); % number of data points
    N = size(data,2); % number of features
    
    result   = zeros(M,1); % each data point will be assigned a cluster
    distance = zeros(M,K); % this will store the distance to each cluster center
    centres  = start_centres;

    iter = 0;
    while iter < max_iter % Stop if iterations surpass max
        for k = 1:K
            % Calculate distance to each cluster
            distance(:,k) = sum(abs(data - repmat(centres(k,:),M,1)),2);
        end
        
        % Find cluster at minimum distance to each point
        [~,I] = min(distance,[],2);

        if I == result % Stop if clusters don't change
            break
        end
        
        % Calculate new centers: find center of points
        
        % counter = zeros(K,1);
        
        for k = 1:K
           pointsincluster = find(I'==k); 
           if ~isempty(pointsincluster)
                centres(k,:) = mean(data([pointsincluster], :), 1);    
           end
        end
        
%         for m = 1:M % for each data point
%             cluster = I(m);
%             centres(cluster,:) = centres(cluster,:) + data(m,:);
%             counter(cluster) = counter(cluster) + 1;
%         end
        
%         for k = 1:K
%             if counter(k) > 0
%                 centres(k,:) = centres(k,:) / counter(k);
%             end
%         end

        result = I;
        iter = iter + 1;
    end
  
end

