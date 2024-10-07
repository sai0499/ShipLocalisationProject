import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utm
from matplotlib.patches import Circle

matplotlib.use('TkAgg')

# Enable interactive mode
plt.ion()

def calc_relative_angle(sensor, ship):
    # sensor and ship are 2 x N arrays
    angle = np.mod(np.arctan2(sensor[1, :] - ship[1, :], sensor[0, :] - ship[0, :]), 2 * np.pi)
    return angle

def ellipse(x, C, c, N):
    # x is the center
    # C is the covariance matrix
    # c is the constant
    # N is the number of points
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Length of the main axes
    a = c * np.sqrt(eigenvalues[0])
    b = c * np.sqrt(eigenvalues[1])
    t = np.linspace(0, 2 * np.pi, N + 1)
    z1 = a * np.cos(t)
    z2 = b * np.sin(t)
    X = x.reshape(-1, 1) + eigenvectors @ np.vstack([z1, z2])
    return X

# Sensors' latitude and longitude
sensors_latlon = np.array([
    [59.54421451622937, 59.58296252718586, 59.57576969007426, 59.43593598670043],
    [10.56664697057754, 10.61733019341999, 10.64846570881250, 10.58136009140111]
])

# Convert lat/lon to UTM coordinates
sensors_lat = sensors_latlon[0, :]
sensors_lon = sensors_latlon[1, :]

utm_coords = [utm.from_latlon(lat, lon) for lat, lon in zip(sensors_lat, sensors_lon)]
x = np.array([coord[0] for coord in utm_coords])
y = np.array([coord[1] for coord in utm_coords])

sensors = {'pos': np.array([x, y])}

# Simulation times
dt = 0.001 * 0.05  # 0.05 ms
n_time_steps = int(300 / dt)  # simulate 300 sec
time_steps = dt * np.arange(1, n_time_steps + 1)

num_sensors = sensors['pos'].shape[1]
sensors['last_delta_angles'] = np.zeros(num_sensors)
sensors['peaks'] = np.full(num_sensors, np.nan)
sensors['time_step_of_measurement'] = np.full(num_sensors, np.nan)
sensors['z'] = np.full((num_sensors, n_time_steps), np.nan)
sensors['number_of_peaks_used_for_LS'] = num_sensors

# Ship parameters
ship = {}
ship['pos'] = np.mean(sensors['pos'], axis=1)  # mean of sensor positions
vel = 24 / 1.944  # Convert knots to m/s
heading = np.deg2rad(-140)
ship['vel'] = np.array([np.cos(heading), np.sin(heading)]) * vel
ship['trajectory'] = np.zeros((2, n_time_steps))
ship['radar'] = {}
ship['radar']['vel'] = -2 * np.pi * 0.5  # -2 * pi * 0.5
ship['radar']['length'] = 8000
ship['radar']['angles'] = np.mod((time_steps - dt) * ship['radar']['vel'], 2 * np.pi)

# Add constant turn rate
ship['turn_rate'] = 0 * dt  # degrees per second
ship['turn_R'] = np.array([[np.cos(np.deg2rad(ship['turn_rate'])), -np.sin(np.deg2rad(ship['turn_rate']))],
                           [np.sin(np.deg2rad(ship['turn_rate'])), np.cos(np.deg2rad(ship['turn_rate']))]])

# Plotting setup
fig, ax = plt.subplots()
ax.set_xlim([np.min(sensors['pos'][0, :]) - 500, np.max(sensors['pos'][0, :]) + 500])
ax.set_ylim([np.min(sensors['pos'][1, :]) - 500, np.max(sensors['pos'][1, :]) + 500])
ax.plot(sensors['pos'][0, :], sensors['pos'][1, :], '^r', markersize=5)

# Add Legend
qw1, = ax.plot(np.nan, np.nan, '--k')
qw2, = ax.plot(np.nan, np.nan, '-xr', markersize=5)
qw3, = ax.plot(np.nan, np.nan, color='#0072BD', linewidth=1, linestyle='-')
ax.legend([qw1, qw2, qw3], ['True Position', 'Least Squares', 'Particle Filter'], loc='upper left', frameon=True)

# System Model
Ax = np.array([[1, dt], [0, 1]])
A = np.kron(Ax, np.eye(2))

# Sensor noise and Process noise
R = 0.1 / 10000  # Variance of R in ms
za = 0.05
Qx = np.array([[1/3*dt**2, 1/2*dt], [1/2*dt, 1]]) * za * dt
Q = np.kron(Qx, np.eye(2))

h_ship_trajectory, = ax.plot([], [], '--k')

# Initialize LS estimate
estLS = {}
estLS['x'] = np.zeros(2)
estLS['P'] = np.zeros((2, 2))
estLS['trajectory'] = np.full((2, n_time_steps), np.nan)
estLS['k'] = -1
h_estLS_trajectory, = ax.plot([], [], '-r', linewidth=0.25)

# Initialize Particle Filter estimate
estPF = {}
estPF['N_particles'] = 1000  # Number of particles
estPF['particles'] = None     # Will be initialized later
estPF['weights'] = None       # Will be initialized later
estPF['trajectory'] = np.full((2, n_time_steps), np.nan)
estPF['k'] = -1

estPF['A'] = A
estPF['Q'] = Q
estPF['R'] = 2 * R

# Initialization parameters for the PF
estPF['initN'] = 7
estPF['initX'] = []
estPF['initk'] = []

# For visualization
h_estPF_trajectory, = ax.plot([], [], '-b', linewidth=0.5)
h_particles_plot = None  # Will hold the particle scatter plot

# Main simulation loop
for k in range(n_time_steps):
    # Ship movement
    ship['pos'] = A[0:2, 0:2] @ ship['pos'] + A[0:2, 2:4] @ ship['vel']
    ship['trajectory'][:, k] = ship['pos']
    ship['vel'] = ship['turn_R'] @ ship['vel']

    # Prediction Step of Particle Filter
    if estPF['particles'] is not None:
        # Predict particles
        # Add process noise
        process_noise = np.random.multivariate_normal(np.zeros(4), estPF['Q'], estPF['N_particles'])
        estPF['particles'] = (estPF['particles'] @ estPF['A'].T) + process_noise

    # Ship radar beam end position
    ship['beam_end'] = np.zeros(2)
    ship['beam_end'][0] = ship['pos'][0] + ship['radar']['length'] * np.cos(ship['radar']['angles'][k])
    ship['beam_end'][1] = ship['pos'][1] + ship['radar']['length'] * np.sin(ship['radar']['angles'][k])

    # Calculate relative angles
    sensor_angles = calc_relative_angle(sensors['pos'], np.tile(ship['pos'], (num_sensors, 1)).T)
    delta_angles = sensor_angles - ship['radar']['angles'][k]

    # Check which sensor detects the radar
    condition = (np.sign(delta_angles) != np.sign(sensors['last_delta_angles'])) & \
                (np.abs(delta_angles - sensors['last_delta_angles']) < 1)
    if np.any(condition):
        # IDs of activated sensors
        id_active_sensor = np.where(condition)[0]
        # Mark active sensor in plot
        if 'h_marked_sensor' in locals():
            h_marked_sensor.remove()
        h_marked_sensor, = ax.plot(sensors['pos'][0, id_active_sensor], sensors['pos'][1, id_active_sensor],
                                   '.', color='#77AC30', markersize=20)

        # Measuring standard deviation with respect to time pogression
        print(f'{k*dt:.2f} sec: Sensor {id_active_sensor} hit! Delta angle: {delta_angles[id_active_sensor]*180/np.pi}')

        # Measurement
        sensors['peaks'][id_active_sensor] = dt * k + np.random.normal(0, np.sqrt(R), size=len(id_active_sensor))
        sensors['time_step_of_measurement'][id_active_sensor] = k

        # If all sensors have detected the ship once
        if np.sum(~np.isnan(sensors['peaks'])) == num_sensors:
            # Remove old circles in plot
            if 'h_circles' in locals():
                for hc in h_circles:
                    hc.remove()
                del h_circles

            # Find sensors that have detected the ship before
            sorted_ids = np.argsort(k - sensors['time_step_of_measurement'])

            # Difference between first and last
            sorted_ids = np.append(sorted_ids, sorted_ids[0])

            r_circle = []
            c_circle = []
            h_circles = []
            for n in range(sensors['number_of_peaks_used_for_LS'] - 1):
                id_first_sensor = int(sorted_ids[n])
                first_peak_time = sensors['peaks'][id_first_sensor]
                id_second_sensor = int(sorted_ids[n+1])
                second_peak_time = sensors['peaks'][id_second_sensor]

                # Calculate ship angle
                ship_angle = (first_peak_time - second_peak_time) * np.pi
                ship_angle = np.mod(ship_angle, np.pi)
                pos_B = sensors['pos'][:, id_second_sensor]
                pos_A = sensors['pos'][:, id_first_sensor]

                d_sensors = np.linalg.norm(pos_A - pos_B)
                # Ensure denominator is not zero
                denom = np.sqrt(2 - 2 * np.cos(2 * ship_angle))
                if denom == 0:
                    denom = 1e-10  # small number to avoid division by zero
                r_circ = d_sensors / denom
                r_circle.append(r_circ)

                c_circ_x = pos_A[0] + (r_circ / d_sensors) * ((pos_B[0] - pos_A[0]) * np.sin(ship_angle) -
                                                              (pos_B[1] - pos_A[1]) * np.cos(ship_angle))
                c_circ_y = pos_A[1] + (r_circ / d_sensors) * ((pos_B[0] - pos_A[0]) * np.cos(ship_angle) +
                                                              (pos_B[1] - pos_A[1]) * np.sin(ship_angle))
                c_circle.append([c_circ_x, c_circ_y])

                circle = Circle((c_circ_x, c_circ_y), r_circ, linestyle=':', color=[0.5, 0.5, 0.5],
                                linewidth=1, fill=False)
                ax.add_patch(circle)
                h_circles.append(circle)

            c_circle = np.array(c_circle)
            r_circle = np.array(r_circle)

            # LS update
            estLS['H'] = 2 * np.diff(c_circle, axis=0)
            estLS['z'] = np.diff(np.sum(c_circle ** 2, axis=1)) - np.diff(r_circle ** 2)

            # LS estimation (without weighting as W is not appropriate here)
            estLS['x'] = np.linalg.lstsq(estLS['H'], estLS['z'], rcond=None)[0]
            estLS['k'] += 1
            estLS['trajectory'][:, estLS['k']] = estLS['x']
            ax.plot(estLS['x'][0], estLS['x'][1], 'xr', markersize=5)

            # Aggregate LS estimates for Particle Filter initialization
            if estPF['particles'] is None:
                estPF['initX'].append(estLS['x'])
                estPF['initk'].append(k)

                if len(estPF['initX']) == estPF['initN']:
                    # Convert list to numpy array
                    init_X = np.column_stack(estPF['initX'])

                    # Compute mean and covariance of initial LS estimates
                    mean_pos = np.mean(init_X, axis=1)  # Mean position
                    cov_pos = np.cov(init_X)            # Covariance of position estimates

                    # Assuming zero initial velocity with high uncertainty
                    mean_vel = np.array([0, 0])
                    cov_vel = 100 * np.eye(2)            # High uncertainty for velocity

                    # Combined mean and covariance for position and velocity
                    mean_state = np.concatenate((mean_pos, mean_vel))
                    cov_state = np.block([
                        [cov_pos, np.zeros((2, 2))],
                        [np.zeros((2, 2)), cov_vel]
                    ])

                    # Initialize particles around the aggregated LS estimate
                    estPF['particles'] = np.random.multivariate_normal(
                        mean_state,
                        cov_state,
                        estPF['N_particles']
                    )
                    estPF['weights'] = np.ones(estPF['N_particles']) / estPF['N_particles']
                    estPF['k'] = 0

                    print('Particle Filter Initialized with Aggregated LS Estimates')

            else:
                # Update Particle Filter using the measurement
                id1 = int(sorted_ids[0])
                id2 = int(sorted_ids[1])
                x1 = sensors['pos'][:, id1]
                x2 = sensors['pos'][:, id2]

                # Measurement
                z12 = -(sensors['peaks'][id1] - sensors['peaks'][id2])

                # Define measurement function h(x_p)
                def h_measurement(particles):
                    # particles is array of shape (N_particles, 4)
                    pos = particles[:, 0:2]
                    r1x = np.linalg.norm(pos - x1.reshape(1, -1), axis=1)
                    r2x = np.linalg.norm(pos - x2.reshape(1, -1), axis=1)
                    # Ensure no division by zero
                    denominator = 2 * r1x * r2x
                    denominator[denominator == 0] = 1e-10
                    cos_value = (r1x ** 2 + r2x ** 2 - np.linalg.norm(x1 - x2) ** 2) / denominator
                    cos_value = np.clip(cos_value, -1.0, 1.0)
                    # Expected time difference
                    expected_z12 = np.arccos(cos_value) / ship['radar']['vel']
                    return expected_z12

                # Compute expected measurements for all particles
                expected_z12 = h_measurement(estPF['particles'])

                # Compute weights using Gaussian likelihood
                measurement_noise_std = np.sqrt(estPF['R'])
                likelihoods = (1 / (np.sqrt(2 * np.pi) * measurement_noise_std)) * \
                              np.exp(-0.5 * ((z12 - expected_z12) / measurement_noise_std) ** 2)

                # Update weights
                estPF['weights'] *= likelihoods
                estPF['weights'] += 1e-300  # Avoid zeros
                estPF['weights'] /= np.sum(estPF['weights'])

                # Effective sample size
                N_eff = 1. / np.sum(estPF['weights'] ** 2)
                N_threshold = estPF['N_particles'] / 2

                # Resample if necessary
                if N_eff < N_threshold:
                    cumulative_sum = np.cumsum(estPF['weights'])
                    cumulative_sum[-1] = 1.  # Avoid round-off error
                    indexes = np.searchsorted(cumulative_sum, np.random.rand(estPF['N_particles']))

                    # Resample according to indexes
                    estPF['particles'] = estPF['particles'][indexes]
                    estPF['weights'] = np.ones(estPF['N_particles']) / estPF['N_particles']

                # Estimate state
                est_state = np.average(estPF['particles'], weights=estPF['weights'], axis=0)
                estPF['k'] += 1
                estPF['trajectory'][:, estPF['k']] = est_state[0:2]

                # Compute and print standard deviation
                # 1. Compute the weighted mean position
                mean_pos_pf = np.average(estPF['particles'][:, 0:2], axis=0, weights=estPF['weights'])

                # 2. Compute the weighted covariance matrix
                diff = estPF['particles'][:, 0:2] - mean_pos_pf
                cov_pos_pf = np.dot(estPF['weights'] * diff.T, diff)

                # 3. Compute eigenvalues of the covariance matrix
                eigenvalues_pf, _ = np.linalg.eigh(cov_pos_pf)

                # 4. Compute standard deviations
                std_devs_pf = np.sqrt(eigenvalues_pf)

                # 5. Compute maximum standard deviation
                max_std_pf = np.max(std_devs_pf)

                # 6. Print the maximum standard deviation
                print(f'[PF] Max. standard deviation in position: {max_std_pf:.2f} m')

                # #*****for Plotting the graph*************#
                # # Configure logging
                # logging.basicConfig(filename='pf_std_dev.log', level=logging.INFO,
                #                     format='%(asctime)s:%(levelname)s:%(message)s')
                #
                # # Replace the print statement with logging
                # # Original line:
                # # print(f'[PF] Max. standard deviation in position: {max_std_pf:.2f} m')
                #
                # # Updated line:
                # logging.info(f'MaxStdDevPF:{max_std_pf:.2f}')
                # # *****for Plotting the graph*************#

                # Plot the estimated position
                ax.plot(est_state[0], est_state[1], '.', color='#0072BD', markersize=7)

    sensors['last_delta_angles'] = delta_angles

    # Update plot every 0.01 seconds
    if k % int(0.01 / dt) == 0:
        if 'h_radar_beam' in locals():
            h_radar_beam.remove()
            h_ship.remove()

        if 'h_marked_sensor' in locals():
            current_size = h_marked_sensor.get_markersize()
            h_marked_sensor.set_markersize(current_size * 0.90)

        # Current position and radar beam
        h_radar_beam, = ax.plot([ship['pos'][0], ship['beam_end'][0]],
                                [ship['pos'][1], ship['beam_end'][1]], color='#77AC30')
        h_ship, = ax.plot(ship['pos'][0], ship['pos'][1], 'ok', markersize=5)

        # Update trajectory
        h_ship_trajectory.set_data(ship['trajectory'][0, :k+1], ship['trajectory'][1, :k+1])

        # Update LS trajectory
        h_estLS_trajectory.set_data(estLS['trajectory'][0, :estLS['k']+1], estLS['trajectory'][1, :estLS['k']+1])

        # Update PF trajectory and particles
        if estPF['particles'] is not None and estPF['k'] >= 0:
            # Remove previous particle plot
            if h_particles_plot is not None:
                h_particles_plot.remove()
            # Plot particles
            h_particles_plot = ax.scatter(estPF['particles'][:, 0], estPF['particles'][:, 1],
                                          c='b', s=1, alpha=0.1)
            # Update PF estimated trajectory
            h_estPF_trajectory.set_data(estPF['trajectory'][0, :estPF['k']+1],
                                        estPF['trajectory'][1, :estPF['k']+1])

        plt.pause(0.001)
        fig.canvas.flush_events()  # Force update of the plot window

# Turn off interactive mode and show the final plot
plt.ioff()
plt.show()
