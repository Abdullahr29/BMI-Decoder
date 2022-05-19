%LSBoost

close all; clear all;

load monkeydata_training.mat

sigma = 2;
sz = 10;    % length of gaussFilter vector
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter); 

yfilt = conv (trial(1,1).spikes(1,:), gaussFilter, 'same');

firing_rate_trials = cell(100, 8);

for j = 1:8
    for k = 1:100
        data_rates = zeros(98, length(trial(k,j).spikes));
        for i = 1:98
            data = trial(k,j).spikes(i,:);
            sigma = 3;
            sz = 15;    % length of gaussFilter vector
            x = linspace(-sz / 2, sz / 2, sz);
            gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
            gaussFilter = gaussFilter / sum (gaussFilter); % normalize
            data_rates(i,:) = conv (data, gaussFilter, 'same');
        end
        firing_rate_trials{k,j} = data_rates;
    end
end

concat_rate_trials = cell(1, 8);
concat_pos_trials = cell(1, 8);

for j = 1:8
    for k = 1:100
        temp_mat = firing_rate_trials{k,j};
        temp_mat = transpose(temp_mat);
        
        temp_pos = trial(k,j).handPos;
        temp_pos = transpose(temp_pos);
        
        if k == 1
            concat_rate_trials{1,j} = temp_mat;
            concat_pos_trials{1,j} = temp_pos;
        else
            concat_rate_trials{1,j} = [concat_rate_trials{1,j};temp_mat];
            concat_pos_trials{1,j} = [concat_pos_trials{1,j};temp_pos];
        end
    end
end

pca_rate_trials = cell(1, 8);

for i = 1:8
    [coeff,score,latent,tsquared,explained,mu] = pca(concat_rate_trials{1,i});
    strc.coeff = coeff;
    strc.variance = explained;
    strc.projection = concat_rate_trials{1,i}*coeff;
    pca_rate_trials{1,i} = strc;
end


%% boost

tic


mdl = fitrensemble(pca_rate_trials{1,1}.projection, concat_pos_trials{1,1}(:,2), 'LearnRate',0.15, 'NumLearningCycles',5000);
toc
t = toc;
%% pred
tic

pred = predict(mdl, pca_rate_trials{1,1}.projection(1:20000, :));

%score =1 - var(concat_pos_trials{1,1}(1:20000,2)-pred)/var(concat_pos_trials{1,1}(1:20000,2));

rmse = sqrt(abs((mean(sum((pred - concat_pos_trials{1,1}(1:20000,2).*concat_pos_trials{1,1}(1:20000,2))) / sum(concat_pos_trials{1,1}(1:20000,2))))));
toc
t=toc;

%% post process

sigma =3;
sz = 14;    % length of gaussFilter vector
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter);

smoothed = conv (pred, gaussFilter, 'same');

figure
hold on
grid on
plot(smoothed, 'linewidth', 1.2)
plot(concat_pos_trials{1,1}(1:20000,2), 'linewidth', 1.2)
xlim([1,2000])
legend('Prediction', 'Ground Truth')
ylabel('Y Position (mm)')
xlabel('Time (ms)')
hold off
truth = concat_pos_trials{1,1}(1:20000,2);

%% post
plot(concat_pos_trials{1,1}(:,2))
