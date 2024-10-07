% Initialize a web map using OpenStreetMap as the base layer
wm = webmap('OpenStreetMap');

% Define the ship's path (latitudes and longitudes)
latitudes = [30.0, 30.2, 30.4, 30.6, 30.8]; % Example latitudes
longitudes = [-90.0, -90.2, -90.4, -90.6, -90.8]; % Example longitudes


% Generate 60 new items
new_latitudes = linspace(30.0, 36.0, 60); % Example range for new latitudes
new_longitudes = linspace(-90.0, -96.0, 60); % Example range for new longitudes

% Append new items to existing arrays
latitudes = [latitudes, new_latitudes];
longitudes = [longitudes, new_longitudes];

% Plot the initial position of the ship on the map
shipMarker = wmmarker(latitudes(1), longitudes(1), 'FeatureName', 'Ship', 'Color', 'red');

% Plot the entire path of the ship on the map (if needed)
wmline(latitudes, longitudes, 'Color', 'blue', 'Width', 2);

% Simulate the movement of the ship
for i = 2:length(latitudes)
    % Remove the old marker
    wmremove(shipMarker);
    
    % Place a new marker at the current location
    shipMarker = wmmarker(latitudes(i), longitudes(i), 'FeatureName', 'Ship', 'Color', 'red');
    
    % Pause for a moment to simulate movement
    pause(1); % Pause time can be adjusted to control movement speed
end
