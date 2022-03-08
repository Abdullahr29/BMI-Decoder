clear all
clc

% DEMO: effects of optimising code by preallocating var space and
% vectorising operations

% Example: you have J timeseries of length I and you want a new smoother
% array where the value at every timepoint i is replaced with the average
% of itself with its 4 temporal neighbours i-2, i-1, i+1, i+2 for every
% timeseries j. The new array will be of length I-2 but we'll pad the the
% missing values with 0s such that it is the same size as the original.

% PLEASE NOTE THAT IS INTENDED AS A DEMO/TUTORIAL ON OPTIMISING CODE BY
% VECTORISING OPERATIONS (AVOIDING FOR LOOPS), NOT AS A SUGGESTION FOR A 
% PREPROCESSING STEP. PLEASE NOTE THAT A BUILT-IN MATLAB FUNCTION IS
% AVAILABE AND FASTER IN THIS CASE AND THAT A BUILT-IN MATLAB FUNCTION IS
% GOING TO BE AVAILABLE IN 99% OF CASES AND FASTER IN 95% OF CASES.

% too big is boring: this demo takes about 15s to run on my machine with
% these settings and I don't recommend wasting more time than this...
J = 22*60; % number of timeseries (channels*trials)
I = 1000*2; % length of timeseries (2 seconds sampled at 1kHz)
% random binary (spike or no spike) timseries
a = randi([0 1], I, J);

% non vectorised, non preallocated operation
tic
for i = 3:size(a,1)-2
    for j = 1:size(a,2)
        b1(i,j) = mean(a(i-2:i+2,j));
    end
end
for j = 1:size(a,2)
    b1(i+1:i+2,j) = 0;
end
t1 = toc;
disp('Non vectorised, non preallocated operation')
disp([num2str(t1, 3), ' s', newline])


% non vectorised, preallocated operation
tic
b2 = zeros(size(a));
for i = 3:size(a,1)-2
    for j = 1:size(a,2)
        b2(i,j) = mean(a(i-2:i+2,j));
    end
end
t2 = toc;
disp('Non vectorised, preallocated operation')
disp([num2str(t2, 3), ' s (', num2str(100*(t1-t2)/t1, 4), '% faster)', newline])

% vectorised, preallocated operation
tic
b3 = zeros(size(a));

b3(3:end-2,:) = (a(1:end-4,:) + a(2:end-3,:)+ a(3:end-2,:) + a(4:end-1,:) + a(5:end,:))/5;

t3 = toc;
disp('Vectorised, preallocated operation')
disp([num2str(t3, 3), ' s (', num2str(100*(t1-t3)/t1, 4), '% faster)', newline])

% using MATLAB built-in function
tic
b4 = zeros(size(a));

b4(3:end-2,:) = movmean(a,5,'Endpoints','discard');

t4 = toc;
disp('Using MATLAB built-in function')
disp([num2str(t4, 3), ' s (', num2str(100*(t1-t4)/t1, 4), '% faster and ', num2str(100*(t3-t4)/t3, 3), '% faster than our fastest one)', newline])

% check that all results are the same
if all(all(b1==b2 & b2 == b3 & b3 == b4))
    disp("... and they're all the same!")
else
    disp("... but something went wrong...")
end

%plot the result and another one with the average being over 21 samples
figure()
hold on
scatter(1:100, a(1:100))
plot(b1(1:100))

b5 = zeros(size(a));
b5(11:end-10,:) = movmean(a,21,'Endpoints','discard');

plot(b5(1:100))

ylim([-0.5, 1.5])
legend(["binary spike train", "5-point moving average", "21-point moving average"])