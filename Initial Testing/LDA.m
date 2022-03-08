%% PCA Tester
%clear all;close all;clc;
%load train dataset

%% Preprocess
start_idx = 251;
end_idx = 550;

truncated_trial = cell(100,8);
for i = 1:8
    for j = 1:100
        bingbong = zeros(98,975);
        bingbong(:,1:length(trial(j,i).spikes(1,:))) = trial(j,i).spikes(:,:);
        truncated_trial{j,i} = bingbong;
    end
end

pcs = cell(100, 8);

for i = 1:8
    for j = 1:100
        [explained, coeff] = our_pca((truncated_trial{j,i}), 1);
        struc.coeff = coeff;
        struc.var = explained;
        pcs{j,i} = struc;
    end
end

sum_pcs = cell(1,8);
average_pcs = cell(1,8);

for i = 1:8
    sum_pcs{1,i} = zeros(98,98);
    for j = 1:100
        sum_pcs{1,i} = sum_pcs{1,i} + pcs{j,i}.coeff;
    end
    average_pcs{1,i} = sum_pcs{1,i}/100;
end

%% Concatenated trials
concat_trial = cell(1,8);
concat_lengths = zeros(1,8);
concat_lengths_train = zeros(1,8);
concat_lengths_test = zeros(1,8);
concat_train = cell(1,8);
concat_test = cell(1,8);
for i = 1:8
    for j = 1:100
        concat_lengths(1,i) = concat_lengths(1,i) + length(trial(j,i).spikes(1,:));     
    end
    for j = 1:80
        concat_lengths_train(1,i) = concat_lengths_train(1,i) + length(trial(j,i).spikes(1,:));     
    end
    for j = 81:100
        concat_lengths_test(1,i) = concat_lengths_test(1,i) + length(trial(j,i).spikes(1,:));     
    end
end

for i = 1:8
    temp_concat = zeros(98, concat_lengths(1,i));
    temp_idx = 1;
    for j = 1:100
        temp_concat(:,temp_idx:(temp_idx+length(trial(j,i).spikes(1,:))-1)) = trial(j,i).spikes(:,:);
        temp_idx = temp_idx + length(trial(j,i).spikes(1,:));
    end
    concat_trial{1,i} = temp_concat;
end

for i = 1:8
    temp_concat = zeros(98, concat_lengths_train(1,i));
    temp_idx = 1;
    for j = 1:80
        temp_concat(:,temp_idx:(temp_idx+length(trial(j,i).spikes(1,:))-1)) = trial(j,i).spikes(:,:);
        temp_idx = temp_idx + length(trial(j,i).spikes(1,:));
    end
    concat_train{1,i} = temp_concat;
    temp_concat = zeros(98, concat_lengths_test(1,i));
    temp_idx = 1;
    for j = 81:100
        temp_concat(:,temp_idx:(temp_idx+length(trial(j,i).spikes(1,:))-1)) = trial(j,i).spikes(:,:);
        temp_idx = temp_idx + length(trial(j,i).spikes(1,:));
    end
    concat_test{1,i} = temp_concat;
end

%% Concat PCA

concat_pcs = cell(1,8);
concat_pcvars = cell(1,8);
for i = 1:8
    [explained, coeff] = our_pca((concat_trial{1,i}), 1);
    concat_pcs{1,i} = coeff;
    concat_pcvars{1,i} = explained;
    
end

%% Concatenated PCs

concat_all_pcs = cell(1,8);

for i = 1:8
    temp_concat = zeros(98, 9800);
    temp_idx = 1;
    for j = 1:100
        temp_concat(:,temp_idx:temp_idx+98-1) = pcs{j,1}.coeff;
        temp_idx = temp_idx + 98;
    end
    concat_all_pcs{1,i} = temp_concat;
end


%% LDA Matlab

%% Full dataset prep
%concat_trial{1,1} = transpose(concat_trial{1,1});

% For classify function need to pass in a test set, training set, and
% classes for all training set data, need to concatenate everything as rows
% of samples and columns of features

train_length = sum(concat_lengths_train);
training_set = zeros(train_length, 98);
class_set = zeros(train_length, 1);

temp_idx = 1;
for i = 1:8
    temp_length = length(concat_train{1,i}(1,:));
    training_set(temp_idx:temp_idx+temp_length-1, :) = transpose(concat_train{1,i});
    class_set(temp_idx:temp_idx+temp_length-1, :) = i;
    temp_idx = temp_idx + temp_length;
end

test_length = sum(concat_lengths_test);
test_set = zeros(test_length, 98);
class_test_set = zeros(test_length, 1);
temp_idx = 1;
for i = 1:8
    temp_length = length(concat_test{1,i}(1,:));
    test_set(temp_idx:temp_idx+temp_length-1, :) = transpose(concat_test{1,i});
    class_test_set(temp_idx:temp_idx+temp_length-1, :) = i;
    temp_idx = temp_idx + temp_length;
end

%% PCA prep

training_pcs = zeros(784, 98);
pcs_class = zeros(784,1);
for i = 1:8
    training_pcs(1+(i-1)*98:98+(i-1)*98,:) = concat_pcs{1,i}(:,:);
    pcs_class(1+(i-1)*98:98+(i-1)*98,:) = i;
end

testing_pcs = zeros(784, 98);
pcs_test_class = zeros(784,1);
for i = 1:8
    [~, coeff]= our_pca(concat_test{1,i},0);
    testing_pcs(1+(i-1)*98:98+(i-1)*98,:) = coeff;
    pcs_test_class(1+(i-1)*98:98+(i-1)*98,:) = i;
end


%% Classify
%So it was recommended to initially try LDA on the raw data, before dim
%reduction, however the following lines do not work:
%https://stackoverflow.com/questions/5923726/matlab-bug-with-linear-discriminant-analysis/16057976#16057976
%link above shows why
%uncomment code to test
%class = classify(test_set,training_set,class_set);
%cm = confusionchart(class_test_set,class);

%using pcs for each class for the training data instead
%class = classify(test_set,training_pcs,pcs_class);
%cm = confusionchart(class_test_set,class);
%conclusion: dogshit

%using pcs for both training and test set
class = classify(testing_pcs,training_pcs,pcs_class);
cf = confusionchart(pcs_test_class,class);
%still dogshit, me stumped, will do some more research


%% OUR LDA: Mean and standardisation

mean_centred = cell(1,8);
class_means = zeros(8,98);

for i = 1:8
   mn = mean(transpose(concat_trial{1,i}));
   %st = std(transpose(concat_trial{1,i}));
   %mean_centred{1,i} = (transpose(concat_trial{1,i}) - mn)./st;
   class_means(i,:) = mn;
end

mn = mean(class_means);
st = std(class_means);

for i = 1:8
   mean_centred{1,i} = (transpose(concat_trial{1,i}) - mn)./st;
end

%% Class and overall means

class_means = zeros(8,98);
for i = 1:8
   mn = mean(mean_centred{1,i});
   class_means(i,:) = mn;
end

