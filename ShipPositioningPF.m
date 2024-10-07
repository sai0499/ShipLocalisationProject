% Circular Trilateration for Ship Position Estimation and Particle Filter Localization
clear
close all
format long g

% Sensor positions (latitude and longitude)
sensors_latlon = horzcat(...
    [59.54421451622937, 10.56664697057754]', ...
    [59.58296252718586, 10.61733019341999]', ...
    [59.57576969007426, 10.64846570881250]', ...
    [59.43593598670043, 10.58136009140111]'...
    );

% Convert to UTM coordinates
z1 = utmzone(sensors_latlon');
[ellipsoid,estr] = utmgeoid(z1);
utmstruct = defaultm('utm');
utmstruct.zone = z1;
utmstruct.geoid = ellipsoid;
utmstruct = defaultm(utmstruct);
[x,y] = projfwd(utmstruct, sensors_latlon(1,:), sensors_latlon(2,:));
sensors.pos = [x;y];

% Simulation times
dt = 0.001 * 0.05; % 0.05 ms
n_time_steps = 300 / dt; % simulate 300 sec
time_steps = dt * (1:n_time_steps);

% Ship initial state
ship.pos = mean(sensors.pos, 2);  % Start in the center of sensors
vel = 24 / 1.944; % 24 knots in m/s
heading = deg2rad(-140); 
ship.vel = [cos(heading); sin(heading)] * vel;
ship.trajectory = zeros(2, n_time_steps);
ship.radar.vel = -2 * pi * 0.5; % 30 rpm
ship.radar.length = 8000;
ship.radar.angles = mod((time_steps - dt) * ship.radar.vel, 2 * pi);
ship.turn_rate = 0 * dt;
ship.turn_R = [cosd(ship.turn_rate), -sind(ship.turn_rate); sind(ship.turn_rate), cosd(ship.turn_rate)];

% Sensor setup
sensors.last_delta_angles = zeros(1, length(sensors.pos));
sensors.peaks = nan(length(sensors.pos), 1);
sensors.time_step_of_measurement = nan(length(sensors.pos), 1);

% Visualization setup
fig = figure(1);
xlim([min([sensors.pos(1,:)]) - 500, max([sensors.pos(1,:)]) + 500])
ylim([min([sensors.pos(2,:)]) - 500, max([sensors.pos(2,:)]) + 500])
hold on
plot([sensors.pos(1,:)], [sensors.pos(2,:)], '^r', 'MarkerSize', 5);
h_ship_trajectory = plot(ship.pos(1), ship.pos(2), '--k');

% Variables to store detection times and distances
detection_times = [];
detected_sensors = [];
detected_distances = [];

% List to store the last 5 estimated positions
estimated_positions = [];
total_estimates_count = 0; % Keep track of total number of estimates

% Particle filter setup
pf_initialized = false;
n_particles = 1000; % Number of particles
particles = []; % Store particles, now including [x, y, vx, vy]
weights = []; % Store particle weights
particle_trajectory = []; % Store trajectory of particle filter estimated state
h_particles = []; % Particle plot handle

% Handles for plotting
h_particle_filter_state = [];
h_circular_trilateration_estimate = [];
h_particle_filter_trajectory = plot(nan, nan, '-b', 'LineWidth', 1); % Initialize trajectory plot

% Simulation loop
for k = 1:n_time_steps
    % Update ship's position and velocity
    ship.pos = ship.pos + ship.vel * dt;
    ship.trajectory(:,k) = ship.pos;
    ship.vel = ship.turn_R * ship.vel;

    % Radar beam endpoint
    ship.beam_end(1) = ship.pos(1) + ship.radar.length * cos(ship.radar.angles(k));
    ship.beam_end(2) = ship.pos(2) + ship.radar.length * sin(ship.radar.angles(k));

    % Calculate relative angles between ship and sensors
    sensor_angles = calc_relative_angle(sensors.pos, repmat(ship.pos, 1, length(sensors.pos)));
    delta_angles = sensor_angles - ship.radar.angles(k);

    % Check for radar hits (sensor detection)
    if any(sign(delta_angles) ~= sign(sensors.last_delta_angles) & (abs(delta_angles - sensors.last_delta_angles) < 1))
        id_active_sensor = find(sign(delta_angles) ~= sign(sensors.last_delta_angles) & (abs(delta_angles - sensors.last_delta_angles) < 1));
        if exist('h_marked_sensor', 'var')
            delete(h_marked_sensor);
        end
        h_marked_sensor = plot(sensors.pos(1, id_active_sensor), sensors.pos(2, id_active_sensor), '.', 'Color', '#77AC30', 'MarkerSize', 80);
        fprintf('%0.2f sec: Sensor %d hit! Delta angle: %d\n', k * dt, id_active_sensor, delta_angles(id_active_sensor) * 180 / pi);
        sensors.peaks(id_active_sensor) = dt * k;
        sensors.time_step_of_measurement(id_active_sensor) = k;

        % Collect time of detection and sensor index
        detection_times(end+1) = k * dt;
        detected_sensors(:, end+1) = sensors.pos(:, id_active_sensor);
        
        % Calculate distance using d = ω * Δt
        if length(detection_times) > 1
            delta_t = detection_times(end) - detection_times(1); % Time difference from first detection
            d = abs(ship.radar.vel * delta_t); % Distance to the ship
            detected_distances(end+1) = d;
        else
            detected_distances(end+1) = 0; % For the first sensor hit
        end
        
        % Perform Circular Trilateration using the function after at least 3 sensors have detected
        if length(detected_distances) >= 3
            estimated_position = estimate_position_via_trilateration(detected_sensors, detected_distances);
            
            % Add the estimated position to the list
            estimated_positions(:, end+1) = estimated_position;
            total_estimates_count = total_estimates_count + 1; % Increment the total estimate count
            
            % Keep only the last 5 positions for visualization
            if size(estimated_positions, 2) > 5
                estimated_positions(:, 1) = [];
            end
            
            % Plot the circular trilateration estimated position
            if exist('h_circular_trilateration_estimate', 'var')
                delete(h_circular_trilateration_estimate);
            end
            h_circular_trilateration_estimate = plot(estimated_positions(1,end), estimated_positions(2,end), 'x', 'MarkerSize', 5, 'Color', 'b'); % 'x' for circular trilateration
            
            fprintf('Estimated Position (Trilateration): X = %f, Y = %f\n', estimated_position(1), estimated_position(2));
            
            % Initialize particle filter after 7 total trilateration estimations
            if total_estimates_count == 10 && ~pf_initialized
                [particles, weights] = initialize_particle_filter(estimated_position, n_particles);
                pf_initialized = true;
                fprintf('Particle Filter Initialized.\n');
                
                % Plot particles
                h_particles = plot(particles(1,:), particles(2,:), '.r');
            end
            
            % Update particle filter if initialized
            if pf_initialized
                [particles, weights] = update_particle_filter(particles, weights, estimated_position, dt);
                
                % Plot particles
                if exist('h_particles', 'var')
                    delete(h_particles); % Remove old particles
                end
                h_particles = plot(particles(1,:), particles(2,:), '.r'); % Plot updated particles
                
                % Estimate state from the particle filter (mean of particle positions)
                particle_filter_state = mean(particles(1:2,:), 2);
                
                % Add to particle trajectory
                particle_trajectory(:, end+1) = particle_filter_state;
                
                % Plot the particle filter estimated state (blue dot of size 5)
                if exist('h_particle_filter_state', 'var')
                    delete(h_particle_filter_state);
                end
                h_particle_filter_state = plot(particle_filter_state(1), particle_filter_state(2), 'o', 'MarkerSize', 5, 'Color', 'b');
                
                % Plot the particle filter trajectory
                set(h_particle_filter_trajectory, 'XData', particle_trajectory(1,:), 'YData', particle_trajectory(2,:));
            end
        end
    end

    sensors.last_delta_angles = delta_angles;

    % Update visualization every 10 ms
    if mod(k * dt, 0.01) == 0
        if exist('h_radar_beam', 'var')
            delete(h_radar_beam);
            delete(h_ship);
        end
        if exist('h_marked_sensor', 'var')
            h_marked_sensor.MarkerSize = h_marked_sensor.MarkerSize * 0.90;
        end
        h_radar_beam = plot([ship.pos(1), ship.beam_end(1)], [ship.pos(2), ship.beam_end(2)], 'Color', '#77AC30');
        h_ship = plot(ship.pos(1), ship.pos(2), 'ok', 'MarkerSize', 5);
        h_ship_trajectory.XData = ship.trajectory(1, 1:k);
        h_ship_trajectory.YData = ship.trajectory(2, 1:k);
        drawnow;
        pause(0.01);
    end
end
hold off

% Function to calculate the relative angle between sensor and ship
function angle = calc_relative_angle(sensor, ship)
    angle = mod(atan2(sensor(2,:) - ship(2,:), sensor(1,:) - ship(1,:)), 2 * pi);
end

% Function for Circular Trilateration to estimate the ship's position
function estimated_position = estimate_position_via_trilateration(detected_sensors, detected_distances)
    % Objective function to minimize the sum of squared differences between
    % the distances to the sensors and the actual distances from the estimated position
    obj_fun = @(x) sum((sqrt((detected_sensors(1,:) - x(1)).^2 + (detected_sensors(2,:) - x(2)).^2) - detected_distances).^2);
    
    % Initial guess for ship position (midpoint of sensors)
    initial_guess = mean(detected_sensors, 2);
    
    % Use fminsearch to minimize the objective function
    estimated_position = fminsearch(obj_fun, initial_guess);
end

% Function to initialize the particle filter
function [particles, weights] = initialize_particle_filter(init_position, n_particles)
    % Initialize particles around the initial position with [x, y, vx, vy]
    particles = repmat([init_position; 0; 0], 1, n_particles) + randn(4, n_particles) * 100; % Add noise to position and velocity
    weights = ones(1, n_particles) / n_particles; % Initialize weights uniformly
end

% Function to update the particle filter
function [particles, weights] = update_particle_filter(particles, weights, measurement, dt)
    % Update particle state (prediction step)
    % particles = [x; y; vx; vy] for each particle
    particles(1,:) = particles(1,:) + particles(3,:) * dt; % Update x position
    particles(2,:) = particles(2,:) + particles(4,:) * dt; % Update y position
    
    % Resample particles based on measurement (update step)
    n_particles = size(particles, 2);
    for i = 1:n_particles
        % Calculate the likelihood of each particle based on its position (x, y) compared to the measurement
        weights(i) = exp(-norm(particles(1:2,i) - measurement)^2 / 100); % Gaussian likelihood based on position
    end
    
    % Normalize weights
    if sum(weights) == 0
        weights = ones(1, n_particles) / n_particles; % Reset weights to avoid division by zero
    else
        weights = weights / sum(weights); % Normalize weights
    end
    
    % Systematic Resampling
    indices = systematic_resample(weights);
    particles = particles(:, indices); % Resample particles based on weights
    weights = ones(1, n_particles) / n_particles; % Reset weights after resampling
end

% Function for systematic resampling
function indices = systematic_resample(weights)
    n_particles = length(weights);
    indices = zeros(1, n_particles);
    cdf = cumsum(weights);
    r = rand / n_particles;
    j = 1;
    for i = 1:n_particles
        u = r + (i-1) / n_particles;
        while u > cdf(j)
            j = j + 1;
        end
        indices(i) = j;
    end
end
