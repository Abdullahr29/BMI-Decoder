%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  
  design_mat = calculate_design_matrix(training_data, 100);

% standarise the design_matrix
    m = mean(design_mat,1);
    s = std(design_mat,1);
    design_mat_standarised = (design_mat - m)./s;

    % just in case
    design_mat_standarised(isnan(design_mat_standarised)) = 0;
    design_mat_standarised(isinf(design_mat_standarised)) = 0;

    [ eigenvalues, principal_components] = our_pca(design_mat_standarised, 1,100);
    % with pc's = 10, time = 0.353711 and there are two errors
    % with pc's = 100, time = 0.358883 and there are no errors
    design_mat_standarised = design_mat_standarised*principal_components;
    Y =repmat([1:1:8]',80,1);
   
    modelParameters = principal_components;
    
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

