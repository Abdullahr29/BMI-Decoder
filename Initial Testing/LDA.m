%% LDA Tester

start_idx = 251;
end_idx = 550;

truncated_trial = cell(100,8);
for i = 1:8
    for j = 1:100
        truncated_trial{j,i} = trial(j,i).spikes(:, 251:550);
    end
end

pcs = cell(100, 8);

for i = 1:8
    for j = 1:100
        [coeff,score,latent,tsquared,explained,mu] = pca(transpose(truncated_trial{j,i}));
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


