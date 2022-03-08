%% Find size of arrays
minlen = 1000;
min_trial = [0,0];
for i = 1:100
    for j = 1:8
        len = length(trial(i,j).spikes(1,:));
        if len<minlen
            minlen = len;
            min_trial = [i,j];
        end
    end
end

% minlen = 571
% maxlen = 975;
%%
clear all
load('monkeydata_training.mat')

%% Create input data
% To perform separate ica on each neuron 
train = zeros(600,98,975); % Leave some data for test
% neurons, 3/4 trials (for 8 angles), time bins
test  = zeros(200,98,975); 

for n = 0:97
    count_train = 1;
    count_test  = 1;

    for j = 0:7
        % Add 75 trials from each direction to the training data
        for i = 1:75
            len = length(trial(i,j+1).spikes);
            train(count_train,n+1,1:len) = trial(i,j+1).spikes(n+1,:);
            count_train = count_train + 1;
        end
        % Add 25 trials from each direction to the testing data
        for i = 76:100
            len = length(trial(i,j+1).spikes);
            test(count_test,n+1,1:len) = trial(i,j+1).spikes(n+1,:);
            count_test = count_test + 1;
        end
    end
end

%% ICA + Kmeans
N = 50; % Number of PCs
transform = my_ica(N,train,100);
cluster(N, transform, train(:,:,1:300), test(:,:,1:300));

%% PCA + Kmeans
N = 50; % Number of PCs
transform = my_pca(N,train);
cluster(N, transform, train, test);

%%
function transform = my_ica(N,train,limit)
    % Perform ICA. Obtain transformations (98,600,975)
    transform = zeros(975,98,N);
    for i=1:975
        res = rica(squeeze(train(:,:,i)),N,'IterationLimit',limit); % Limit iterations to make faster!
        transform(i,:,:) = res.TransformWeights;
    end  
end

function transform = my_pca(N,train)
    % Perform PCA. Obtain transformations
    transform = zeros(975,98,N);
    for i=1:975
        transform(i,:,:) = pca(squeeze(train(:,:,i)),'NumComponents',N);
    end  
end

function cluster(N,transform,train,test)
    % Transform data using ica/pca principal components
    transformed_train = zeros(600,975,N); 
    transformed_test  = zeros(200,975,N); 
    
    for i=1:975
        transformed_train(:,i,:) = squeeze(train(:,:,i)) * squeeze(transform(i,:,:));
        % 600*98 x 98*50
        transformed_test(:,i,:)  = squeeze(test(:,:,i))  * squeeze(transform(i,:,:));
        % 200*98 x 98*50
    end 
    
    % Get mean signal per time and per direction
    average_per_time = zeros(975,8,N);
    for i = 0:7
        for j = 1:975
            average_per_time(j,i+1,:) = squeeze(mean(transformed_train((i*75)+1:(i+1)*75,j,:)));
        end
    end

    % Use K-means to cluster results
    % Use averages as starting cluster centers
    idx_train = zeros(600,975);
    idx_test  = zeros(200,975);
    
    for i = 1:975
        idx_train(:,i) = kmeans(squeeze(transformed_train(:,i,:)),8,'Start',squeeze(average_per_time(i,:,:)));
        idx_test(:,i)  = kmeans(squeeze(transformed_test(:,i,:)) ,8,'Start',squeeze(average_per_time(i,:,:)));
    end
    res_train = mode(idx_train);  x_train = 1:600;
    res_test  = mode(idx_test);   x_test  = 1:200;

    % Plot results
    figure()
    for i = 0:7
        subplot(1,2,1); hold on; plot(x_train(i*75+1:(i+1)*75), res_train(i*75+1:(i+1)*75)); 
        title('Training data'); xlabel('Trial number'); ylabel('Estimated direction')
        subplot(1,2,2); hold on; plot(x_test(i*25+1:(i+1)*25),  res_test (i*25+1:(i+1)*25)); 
        title('Test data'); xlabel('Trial number'); ylabel('Estimated direction')
    end
end