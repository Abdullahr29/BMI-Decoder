load('monkeydata_training.mat')
ix = randperm(length(trial));
params = positionEstimatorTraining(trial(ix(1:75),:));

[x,y] = positionEstimator(trial(ix(78),1),params)

% 1.
% Use some data to create a model
% We have to return the characteristic model parameters
  
function average_pcs = positionEstimatorTraining(trainingData)
    data = cell(75,8);
    pcs = cell(75,8);

    for i = 1:8
        for j = 1:75 
            bingbong = zeros(98,975);
            bingbong(:,1:length(trainingData(j,i).spikes(1,:))) = trainingData(j,i).spikes(:,:);
            data{j,i} = bingbong;
        end
    end

    for i = 1:8
        for j = 1:75
            [~, coeff] = our_pca((data{j,i}), 1, 80);
            pcs{j,i} = coeff;
        end
    end

    sum_pcs = cell(1,8);
    average_pcs = cell(1,8);
 
    for i = 1:8
        sum_pcs{1,i} = zeros(98,80); % change dimensions here too !!
        for j = 1:75
            sum_pcs{1,i} = sum_pcs{1,i} + pcs{j,i};
        end
        average_pcs{1,i} = sum_pcs{1,i}/100;
    end
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


function [posX, posY] = positionEstimator(trial, params)
    data = zeros(98,975);
    data(:,1:length(trial.spikes(1,:))) = trial.spikes(:,:); 
    [~, coeff] = our_pca(data, 1, 80);

    error = zeros(8,98,80);
    for i = 1:8
       error(i,:,:) = params{1,i} - coeff;
    end
    
    mean_error = squeeze(mean(error,2));
    mean_error = squeeze(mean(mean_error,2));
    [~,I] = max(mean_error);
    if I == 1
        posX = 67;
        posY = 50;
    end
    if I == 2
        posX = 24;
        posY = 90;
    end
    if I == 3
        posX = -46;
        posY = 88;
    end
    if I == 4
        posX = -93;
        posY = 46;
    end
    if I == 5
        posX = -102;
        posY = -24;
    end
    if I == 6
        posX = -68;
        posY = -80;
    end
    if I == 7
        posX = 58;
        posY = -77;
    end
    if I == 8
        posX = 86;
        posY = -17;
    end

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

