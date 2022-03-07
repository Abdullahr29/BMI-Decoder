% Script with our ICA function and testing it
% code adapted from https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
load('monkeydata_training.mat');

%% Separating data into training and testing (LIV)
train = trial(1:80, :);
test = trial(81:end, :);

% Arranging into array of padded spike trains
train_spikes = zeros(100, 8, 98, 975);
% each trial, each angle, each neuron, time series

for ang = 1:8
    for tri = 1:80
        for neur = 1:98
            n = length(train(tri, ang).spikes(neur,:));
            train_spikes(tri, ang, neur, 1:n) = train(tri, ang).spikes(neur,:);
        end
    end
end

%% ICA: Create input data
% Perform separate ICA on each neuron
train = zeros(98,600,975); % neurons, 3/4 of trials (for 8 angles), time bins
test = zeros(98,200,975); % Leave some data for test

for n = 0:97
    count_train = 1;
    count_test  = 1;

    for j = 0:7
        % Add 75 trials from each direction to the training data
        for i = 1:75
            len = length(trial(i,j+1).spikes);
            train(n+1,count_train,1:len) = trial(i,j+1).spikes(n+1,:);
            count_train = count_train + 1;
        end
        % Add 25 trials from each direction to the testing data
        for i = 76:100
            len = length(trial(i,j+1).spikes);
            test(n+1,count_test,1:len) = trial(i,j+1).spikes(n+1,:);
            count_test = count_test + 1;
        end
    end
end

%% ICA
% Number of ICs is the rows of the input
N = 98;
transform = zeros(600, 98, N);
iterations = 200;
tolerance = 1e-4;

for i =1:600
    transform(i, :, :) = ica(squeeze(train(:, i, :)), iterations, tolerance);
end

cluster(N, transform, train, test);
%% ICA testing: Test Signals
tolerance = 1e-5; % If the new weights vector differs from the old one by less than 'tolerance', we terminate the iteration 

% Generating test signals
n_samples = 2000; % Length of independent sources
time = linspace(0, 8, n_samples); 
s1 = sin(2*time);
s2 = sign(sin(3*time));
s3 = sawtooth(2*pi*time);

% Building mixed signals matrix
X = [s1', s2', s3']; % each column contains one recording 
% (3 columns = want to identify 3 sources)
% hence for our example we want 98 columns?
%X = [s1; s2; s3];
A = [1,1,1; 0.5,2,1; 1.5,1,2];
X = X*A';
%% ICA testing
iterations = 1000;
S = ica(X, iterations, tolerance);
plot_mixture_sources_predictions(X, [s1; s2; s3], S)



% ICA custom functions
% x contains p variables (98)
%X = train_spikes(tri,ang,:,:);
% mdl = rica(X, 8, 'IterationLimit', 500);
%[V, D] = eig(A);

% Hyperbolic tangent
function hyper_tan = g(x)
    hyper_tan = tanh(x);
end

% Derivative of the hyperbolic tangent
function hyper_tan_der = g_der(x)
    hyper_tan_der = 1 - (g(x) .* g(x));
end

% Center the signal (might not be needed in our case?)
function centered = center(X)
    avg = mean(X); % check if you need to specify dimension
    centered = X - avg;
end

% Whitening the signal (remove correlation between components)
function X_whitened = whiten(X)
    % Covariance matrix
    covar = cov(X);
    
    % Eigenvalue decomposition
    [V, D] = eig(covar);
    % D: diagonal matrix of eigenvalues
    % V: orthogonal matrix whose columns correspond to eigenvectors in*V = V*D
    D_inv = sqrt(inv(D));
    
    % Whitened signal
    X_whitened = V *(D_inv * (V' * X')); 
    %X_whitened = V *(D \ (V' * X')); 
    %check_mat = abs(X_whitened_slow - X_whitened);
    %check = max(check_mat);
    %print(check)
end

% Update the demixing matrix
function w_new = update_demixing_mat(w, X)
    a = g(w * X);
    b = [(X(1,:).* a); (X(2,:).*a); (X(3,:).* a)];
    c = mean(b, 2);
    a2 = g_der(w * X); 
    b2 = mean(a2);
    c2 = b2 .* w;
    w_new = c' - c2; %works until here
    %w_new = mean((X * g(w' * X))) - mean(g_der(w'*X) * w);
    %squaroot = sqrt(sum(w_new.^2));
    w_new = w_new/sqrt(sum(w_new .^2));
end

% ICA 
%   Note: would be better trying to implement with mutual information
function sources_est = ica(X, iterations, tolerance)
    % Center
    X = center(X);
    
    % Whiten
    X = whiten(X);
    
    [components_num, ~] = size(X); % rows of X = number of independent components  
    
    W = zeros(components_num);
    
    for i = 1:components_num
       %w = rand(1, components_num);
       w = [0.2, 0.5, 0.3];
       for j = 1:iterations
          w_new = update_demixing_mat(w, X);
          
          if i>1
              w_new = w_new - (w_new * (W(1:i, :)') * W(1:i, :));
          end
          
          distance = abs(abs(sum(w .* w_new))- 1);
          
          % Update 
          w = w_new;
          
          if distance < tolerance
              break
          end
          
       end
       
       W(i, :) = w;
    end
    sources_est = W * X;
end

% Function to plot and compare the original, mixed and predicted signals
function plot_mixture_sources_predictions(X, original_sources, sources_est)
    figure();
    
    subplot(3, 1, 1); hold on
    
    for i = 1:3
        x = X(:, i)';
        plot(x)
    end
    title('mixtures');
    
    subplot(3, 1, 2); hold on
    for i = 1:3
        s = original_sources(i, :);
        plot(s)
    end
    title('real sources')
    
    subplot(3, 1, 3); hold on
    for i = 1:3
        s = sources_est(i, :);
        plot(s)
    end
    title('predicted sources')
end

function cluster(N, transform, train, test)
    % Transform data using ica/pca principal components
    transformed_train = zeros(98,600,N);
    transformed_test  = zeros(98,200,N);
end