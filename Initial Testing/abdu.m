spikes = zeros(100,750);

for i=1:100
    spike = trial(i,1).spikes(1,:);
    lens = length(spike);
    spikes(i,1:lens) = spike;   
end    
figure(1)    
imagesc(spikes)

figure(2)
final_spikes = sum(spikes);
bar(smooth(final_spikes, 1))

hand_x = trial(71,6).handPos(1,:);
hand_y = trial(71,6).handPos(2,:);

t = 1:length(hand_x);

figure(3)
scatter(hand_x, hand_y)

figure(4)
hold on
scatter(t, hand_x)
scatter(t, hand_y)
hold off

hand_x = trial(4,6).handPos(1,:);
hand_y = trial(4,6).handPos(2,:);

t = 1:length(hand_x);

figure(5)
scatter(hand_x, hand_y)

figure(6)
hold on
scatter(t, hand_x)
scatter(t, hand_y)
hold off

hand_x = trial(12,7).handPos(1,:);
hand_y = trial(12,7).handPos(2,:);

t = 1:length(hand_x);

figure(7)
scatter(hand_x, hand_y)

figure(8)
hold on
scatter(t, hand_x)
scatter(t, hand_y)
hold off

figure(9)
vals = zeros(1,8);
for i=1:8
    for j=1:100
        spike = trial(j,i).spikes(1,:);
        lens = length(spike);
        spikes(j,1:lens) = spike;
    end
    final_spikes = mean(spikes);
    final_spikes = mean(final_spikes);
    vals(i) = final_spikes;
end
bar(vals)