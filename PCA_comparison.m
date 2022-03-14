%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPTION 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train = zeros(98,600,500); 
test  = zeros(98,200,500);

for n = 0:97
    count_train = 1;
    count_test  = 1;

    for j = 0:7
        for i = 1:75
            train(n+1,count_train,:) = trial(i,j+1).spikes(n+1,51:550);
            count_train = count_train + 1;
        end
        for i = 76:100
            test(n+1,count_test,:) = trial(i,j+1).spikes(n+1,51:550);
            count_test = count_test + 1;
        end
    end
end

N = 60; % Number of PCs
transform = my_pca(N,train);
cluster(N, transform, train, test);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPTION 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 50;
% zero padding the training data
trainingData = cell(75,8);
testingData = cell(25,8);

for i = 1:8
    for j = 1:75 % gets only the 75 first trials for training
        bingbong = zeros(98,975);
        bingbong(:,1:length(trial(j,i).spikes(1,:))) = trial(j,i).spikes(:,:);
        trainingData{j,i} = bingbong;
    end
    for j = 1:25
        bingbong = zeros(98,975);
        bingbong(:,1:length(trial(j+75,i).spikes(1,:))) = trial(j+75,i).spikes(:,:);
        testingData{j,i} = bingbong;
    end
end

pcs_train = cell(75, 8); all_pcs_train = zeros(75*8,98*N);
pcs_test  = cell(25, 8); all_pcs_test  = zeros(25*8,98*N);

for i = 1:8
    for j = 1:75
        [explained, coeff] = our_pca((trainingData{j,i}), 1, N);
        struc.coeff = coeff; % size = 98xN
        struc.var = explained;
        pcs_train{j,i} = struc;
        all_pcs_train(75*(i-1)+j,:) = reshape(coeff, [1,98*N]);
    end
    for j = 1:25
        [explained, coeff] = our_pca((testingData{j,i}), 1, N);
        struc.coeff = coeff; % size = 98xN
        struc.var = explained;
        pcs_test{j,i} = struc;
        all_pcs_test(25*(i-1)+j,:) = reshape(coeff, [1,98*N]);
    end
end

sum_pcs = cell(1,8);
average_pcs = zeros(8,98*N);
for i = 1:8
    sum_pcs{1,i} = zeros(98,N); 
    for j = 1:75
        sum_pcs{1,i} = sum_pcs{1,i} + pcs_train{j,i}.coeff;
    end
    average_pcs(i,:) = reshape(sum_pcs{1,i}/75, [1,98*N]);
end

idx_train = our_kmeans(all_pcs_train,8,average_pcs,100);
idx_test  = our_kmeans(all_pcs_test ,8,average_pcs,100);
x_train = 1:600; x_test  = 1:200;

% Plot results
figure()
for i = 0:7
    subplot(1,2,1); hold on; plot(x_train(i*75+1:(i+1)*75), idx_train(i*75+1:(i+1)*75)'); 
    title('Training data'); xlabel('Trial number'); ylabel('Estimated direction')
    subplot(1,2,2); hold on; plot(x_test(i*25+1:(i+1)*25),  idx_test (i*25+1:(i+1)*25)'); 
    title('Test data'); xlabel('Trial number'); ylabel('Estimated direction')
end

ideal_train = ones(1,600); ideal_train(76:150) = 2; ideal_train(151:225) = 3; ideal_train(226:300) = 4; 
ideal_train(301:375) = 5; ideal_train(376:450) = 6; ideal_train(451:525) = 7; ideal_train(526:600) = 8;
ideal_test = ones(1,200); ideal_test(26:50) = 2; ideal_test(51:75) = 3; ideal_test(76:100) = 4; 
ideal_test(101:125) = 5; ideal_test(126:150) = 6; ideal_test(151:175) = 7; ideal_test(176:200) = 8;

error_train = sum(ideal_train ~= idx_train') /600
error_test  = sum(ideal_test  ~= idx_test')  /200



%% FUNCTIONS
function transform = my_pca(N,train)
    % Perform PCA. Obtain transformations
    transform = zeros(98,500,N);
    for i=1:98
        transform(i,:,:) = pca(squeeze(train(1,:,:)),'NumComponents',N);
    end  
end

function cluster(N,transform,train,test)
    % Transform data using ica/pca principal components
    transformed_train = zeros(98,600,N); % 98*600*50
    transformed_test  = zeros(98,200,N);% 98*200*50
    
    for i=1:98
        transformed_train(i,:,:) = squeeze(train(i,:,:)) * squeeze(transform(i,:,:));
        % 600*975 x 975*50
        transformed_test(i,:,:)  = squeeze(test(i,:,:))  * squeeze(transform(i,:,:));
        % 200*975 x 975*50
    end 
    
    % Get mean signal per neuron and per direction
    average_per_neuron = zeros(98,8,N);
    for i = 0:7
        for j = 1:98
            average_per_neuron(j,i+1,:) = squeeze(mean(transformed_train(j,(i*75)+1:(i+1)*75,:)));
        end
    end

    % Use K-means to cluster results
    % Use averages as starting cluster centers
    idx_train = zeros(98,600);
    idx_test  = zeros(98,200);
    
    for i = 1:98
        idx_train(i,:) = our_kmeans(squeeze(transformed_train(i,:,:)),8,squeeze(average_per_neuron(i,:,:)),100);
        idx_test(i,:)  = our_kmeans(squeeze(transformed_test(i,:,:)) ,8,squeeze(average_per_neuron(i,:,:)),100);
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

    ideal_train = ones(1,600); ideal_train(76:150) = 2; ideal_train(151:225) = 3; ideal_train(226:300) = 4; 
    ideal_train(301:375) = 5; ideal_train(376:450) = 6; ideal_train(451:525) = 7; ideal_train(526:600) = 8;
    ideal_test = ones(1,200); ideal_test(26:50) = 2; ideal_test(51:75) = 3; ideal_test(76:100) = 4; 
    ideal_test(101:125) = 5; ideal_test(126:150) = 6; ideal_test(151:175) = 7; ideal_test(176:200) = 8;
    
    error_train = sum(ideal_train ~= res_train) /600
    error_test  = sum(ideal_test  ~= res_test)  /200

end