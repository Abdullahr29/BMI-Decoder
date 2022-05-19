% close all; clear all;

%% Calculate Firing Rates
tic
for i = 100:100
    design_mat = calculate_design_matrix(trial, 80);

    % standarise the design_matrix
    m = mean(design_mat,1);
    s = std(design_mat,1);
    design_mat_standarised = (design_mat - m)./s;

    % just in case
    design_mat_standarised(isnan(design_mat_standarised)) = 0;
    design_mat_standarised(isinf(design_mat_standarised)) = 0;



    principal_components = pca(design_mat_standarised,'NumComponents',i);
    % with pc's = 10, time = 0.353711 and there are two errors
    % with pc's = 100, time = 0.358883 and there are no errors
    design_mat_standarised = design_mat_standarised*principal_components;
    Y=repmat([1:1:8]',80,1);
    model = fitcdiscr(design_mat_standarised,Y);

    % prep test data

    design_mat_test = calculate_test_design_matrix(trial, 20);
    % m = mean(design_mat_test,1);
    % s = std(design_mat_test,1);
    design1 = (design_mat_test - m)./s;
    design1(isnan(design1)) = 0;
    design1(isinf(design1)) = 0;
    %[ eigenvalues, principal_components1] = our_pca(design1, 0,20);
    design2 = design1*principal_components;
    Y2=repmat([1:1:8]',20,1);
    % class = classify(design_mat_standarised_test,design_mat_standarised,Y);
    % figure
    % cf = confusionchart(Y2,class);
    class = predict(model,design2); 
    toc
    t = toc;
    figure
    cf = confusionchart(class,Y2);
    rmse(i) = sqrt(mean((Y2 - class).^2));
end

%% kmeans
%close all;
[idx,C] = kmeans(design_mat_standarised,8);

%% process
idx = abs(idx);

[~,main_idx] = max(idx, [], 2);

clusters = reshape(main_idx, 8, 80);

cluster = mode(clusters, 2);


%% Functions
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


function design_mat_test = calculate_test_design_matrix(spike_data, training_size)
    %spike_data: full set of unprocessed spike data
    %training_size: rows out of 100 to use for training and design matrix
    %window_sizes: 1x2 array indicating the size of the prep and after
    %movement windows
    
    
    fr_avg = zeros(training_size*8,98);
    fr_avg_pa = zeros(training_size*8,98);
    fr_avg_ma = zeros(training_size*8,98);
    fr_avg_c = zeros(training_size*8,98);
    temp = 0;
    for i = 81: 80+training_size
        for j = 1:8
            temp = temp + 1;
            fr_avg(temp,:) = average_fr(spike_data(i,j).spikes(:,:));
            fr_avg_pa(temp,:) = average_fr(spike_data(i,j).spikes(:,1:300));
            fr_avg_ma(temp,:) = average_fr(spike_data(i,j).spikes(:,301:end-100));
            fr_avg_c(temp,:) = average_fr(spike_data(i,j).spikes(:,end-99:end));
               
        end
    end
    design_mat_test =[fr_avg,fr_avg_pa,fr_avg_ma,fr_avg_c];
end

function [Y, W, lambda] = our_LDA(X, L)
    Classes=unique(L)';
    k=numel(Classes);
    n=zeros(k,1);
    C=cell(k,1);
    M=mean(X);
    S=cell(k,1);
    Sw=0;
    Sb=0;
    for j=1:k
        Xj=X(L==Classes(j),:);
        n(j)=size(Xj,1);
        C{j}=mean(Xj);
        S{j}=0;
        for i=1:n(j)
            S{j}=S{j}+(Xj(i,:)-C{j})'*(Xj(i,:)-C{j});
        end
        Sw=Sw+S{j};
        Sb=Sb+n(j)*(C{j}-M)'*(C{j}-M);
    end
    [W, LAMBDA]=eig(Sb,Sw);
    lambda=diag(LAMBDA);
    [lambda, SortOrder]=sort(lambda,'descend');
    W=W(:,SortOrder);
    Y=X*W;
end