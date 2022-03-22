function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
    start_centres = zeros(100,1);
  %[result, centres] = our_kmeans(test_data, model_parameters, start_centres, 200);
    [A, T] = size(test_data);
    design_mat_test = calculate_design_matrix(test_data, A);
    m = mean(design_mat_test,1);
    s = std(design_mat_test,1);
    design1 = (design_mat_test - m)./s;
    design1(isnan(design1)) = 0;
    design1(isinf(design1)) = 0;
    %[ eigenvalues, principal_components1] = our_pca(design1, 0,20);
    design2 = design1*modelParameters;
    v = randn(1000);
    x = v(50);
    y = v(888);
   
end

function avg_fr = average_fr(spike_data)
    %spike_data: any matrix of neurons x spikes(over time)
    [neurons, len_data] = size(spike_data);
    avg_fr = zeros(neurons,1);
    avg_fr(:,1) = sum(spike_data,2);
    avg_fr = avg_fr./len_data;
    %avg_fr = transpose(avg_fr);
end



function design_mat = calculate_design_matrix(spike_data, training_size)
    %spike_data: full set of unprocessed spike data
    %training_size: rows out of 100 to use for training and design matrix
    %window_sizes: 1x2 array indicating the size of the prep and after
    %movement windows
    
    
    fr_avg = zeros(training_size*8,98);
    fr_avg_pa = zeros(training_size*8,98);
    fr_avg_ma = zeros(training_size*8,98);
    fr_avg_c = zeros(training_size*8,98);
    temp = 0;
    for i = 1:training_size
        for j = 1:8
            temp = temp + 1;
            fr_avg(temp,:) = average_fr(spike_data(i,j).spikes(:,:));
            fr_avg_pa(temp,:) = average_fr(spike_data(i,j).spikes(:,1:300));
            fr_avg_ma(temp,:) = average_fr(spike_data(i,j).spikes(:,301:end-100));
            fr_avg_c(temp,:) = average_fr(spike_data(i,j).spikes(:,end-99:end));
               
        end
    end
    design_mat =[fr_avg,fr_avg_pa,fr_avg_ma,fr_avg_c];
end

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
        for k = 1:K
           pointsincluster = find(I'==k); 
           if ~isempty(pointsincluster)
                centres(k,:) = mean(data([pointsincluster], :), 1);    
           end
        end

        result = I;
        iter = iter + 1;
    end
  
end