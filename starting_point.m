%% Find average starting point
starts_x = [];
starts_y = [];

for n = 1:8
    for i = 1:100
        position = trial(i,n).handPos(:,1);
        starts_x = [starts_x; position(1)];
        starts_y = [starts_y; position(2)];
    end
end

starting_position = [mean(starts_x),mean(starts_y)];