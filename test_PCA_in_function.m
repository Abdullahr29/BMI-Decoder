% Basically, we have to create 2 functions
 
ix = randperm(length(trial));
params = positionEstimatorTraining(trial(ix(1:75),:));
% zero padding the training data
trainingData = cell(80,8);
for i = 1:8
    for j = 1:80 % gets only the 80 first trials for training
        bingbong = zeros(98,975);
        bingbong(:,1:length(trial(j,i).spikes(1,:))) = trial(j,i).spikes(:,:);
        trainingData{j,i} = bingbong;
    end
end
 
% zero padding testing data
testingData = cell(20,8);
for i = 1:8
    for j = 80:100 % gets only the 20 last trials for testing
        bingbong = zeros(98,975);
        bingbong(:,1:length(trial(j,i).spikes(1,:))) = trial(j,i).spikes(:,:);
        testingData{j-79,i} = bingbong;
    end
end
average_pcs = positionEstimatorTraining(trainingData);
% 1.
% Use some data to create a model
% We have to return the characteristic model parameters
  
function average_pcs = positionEstimatorTraining(trainingData)
    % trainingData is of the form 80x8 struct (each item has trialId,
    % spikes and handPos fields)
    % We will transform spikes data to suit our needs :)
    pcs = cell(80, 8);
   
    for i = 1:8
        for j = 1:80
            [explained, coeff] = our_pca((trainingData{j,i}), 1, 80);
            struc.coeff = coeff;
            struc.var = explained;
            pcs{j,i} = struc;
        end
    end
    sum_pcs = cell(1,8);
    average_pcs = cell(1,8);
 
    for i = 1:8
        sum_pcs{1,i} = zeros(98,80); % change dimensions here too !!
        for j = 1:80
            sum_pcs{1,i} = sum_pcs{1,i} + pcs{j,i}.coeff;
        end
        average_pcs{1,i} = sum_pcs{1,i}/100;
    end
    
    
    
    
    
    
    
%     len = length(trainingData);
%     data = zeros(98,len*8,500); 
%   
%     for n = 0:97
%         count = 1;
%         for j = 0:7
%             for i = 1:len
%                 data(n+1,count,:) = trainingData(i,j+1).spikes(n+1,51:550);
%                 count = count + 1;
%             end
%         end
%     end
% 
%     N = 50; % Number of PCs
%     pca_PCs = zeros(98,len*8,N);
%     for i=1:98
%         [~,PCs] = our_pca(squeeze(data(i,:,:)),0);
%         pca_PCs(i,:,:) = PCs(:,1:N);
%     end  
 
    
    % USE PCA RESULTS TO TRANSFORM DATA!
    % CREATE AVG SIGNAL FOR EACH OF THE 8 DIRECTIONS! 
 
end
 
function [eigenvalues, principal_components] = our_pca(data, variance_explained, dims)
    [U,S,~] = svd(data, "econ");
    eigenvalues = diag(S);
    principal_components = U;
    total = 0;
    
    if(variance_explained == 1)
        for i=1:size(data)
            total = total + eigenvalues(i);
        end
        eigenvalues = (eigenvalues)./total;
        for i=2:size(data)
            eigenvalues(i) =eigenvalues(i-1)+ eigenvalues(i);
        end
    end
    principal_components = principal_components(:,1:dims);
end
 
% 2.
% Estimate position for each input (each combination of trial, direction) 
% Use trained model parameters
 
% Input consists of:
% - trialId 
% - spikes: matrix of shape (98,t), t=320:20:length
% - startHandPos: handPos(1:2,1); 
% Output is the decodedHandPos
 
% Parameters can be updated in estimation
% [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
% [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
 
function [posX, posY] = positionEstimator(trial, params);
    % params: STRUCT = PCs(98x50), Kmeans_centres(8,50)
    % Transform data using ica/pca principal components
    spikes = trial.spikes; % 98 x changing length t
    len    = size(spikes,2);
    
    % USE PCA RESULTS TO TRANSFORM DATA!
 
    % Use K-means to cluster results
    % Use averages as starting cluster centers
    
end




