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

def sigma_points(x, C, w0=1/3):
    # x is the mean state vector
    # C is the covariance matrix
    # w0 is the weight of the central point
    dimx = len(x)
    S = np.sqrt(dimx / (1 - w0)) * np.linalg.cholesky(C)
    T = np.hstack([np.zeros((dimx, 1)), np.eye(dimx), -np.eye(dimx)])
    s = x.reshape(-1, 1) + S @ T
    w = np.hstack([w0, np.ones(2 * dimx) * (1 - w0) / (2 * dimx)])
    return s, w

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
ax.legend([qw1, qw2, qw3], ['true pos.', 'Least Squares', '(U)KF'], loc='upper left', frameon=True)

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

# Initialize KF estimate
estKF = {}
estKF['x'] = np.full(4, np.nan)
estKF['P'] = np.zeros((4, 4))
estKF['trajectory'] = np.full((2, n_time_steps), np.nan)
estKF['k'] = -1

estKF['A'] = A
estKF['Q'] = Q
estKF['R'] = 2 * R

# Initialization parameters for the KF
estKF['initN'] = 7
estKF['initX'] = []
estKF['initZ'] = []
estKF['initH'] = []
estKF['initW'] = []
estKF['initk'] = []

estKF['correlation'] = np.zeros(estKF['x'].shape)

# Main simulation loop
for k in range(n_time_steps):
    # Ship movement
    ship['pos'] = A[0:2, 0:2] @ ship['pos'] + A[0:2, 2:4] @ ship['vel']
    ship['trajectory'][:, k] = ship['pos']
    ship['vel'] = ship['turn_R'] @ ship['vel']

    # Prediction Step of Filters
    if not np.isnan(estKF['x']).any():  # Wait for KF to be initialized
        estKF['x'] = estKF['A'] @ estKF['x']
        estKF['P'] = estKF['A'] @ estKF['P'] @ estKF['A'].T + estKF['Q']
        estKF['k'] = estKF['k'] + 1
        estKF['trajectory'][:, estKF['k']] = estKF['x'][0:2]

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

                c_circ_x = pos_A[0] + r_circ / d_sensors * ((pos_B[0] - pos_A[0]) * np.sin(ship_angle) -
                                                            (pos_B[1] - pos_A[1]) * np.cos(ship_angle))
                c_circ_y = pos_A[1] + r_circ / d_sensors * ((pos_B[0] - pos_A[0]) * np.cos(ship_angle) +
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

            # Initialize KF with first initN LS estimates
            if np.isnan(estKF['x']).any():
                if estKF['initN'] == 1:
                    estKF['x'] = np.concatenate((estLS['x'], [0, 0]))
                    estKF['P'] = 10 * np.block([
                        [np.linalg.inv(estLS['H'].T @ estLS['H']), np.zeros((2, 2))],
                        [np.zeros((2, 2)), 100 * np.eye(2)]
                    ])
                    h_estKF_trajectory, = ax.plot([], [], '-b')
                    estKF['trajectory'][:, 0] = estKF['x'][:2]
                    estKF['k'] = 0
                elif estKF['initN'] == len(estKF['initk']) + 1:
                    estKF['initX'].insert(0, estLS['x'])
                    dk = [0] + [k - k_i for k_i in estKF['initk']]

                    # FIR estimate
                    HFIR = []
                    Hpos = np.hstack([np.eye(2), np.zeros((2, 2))])
                    for i in range(estKF['initN']):
                        HFIR.append(Hpos @ np.linalg.matrix_power(np.linalg.inv(estKF['A']), dk[i]))
                    HFIR = np.vstack(HFIR)

                    zFIR = np.hstack(estKF['initX'])

                    estKF['x'] = np.linalg.lstsq(HFIR.T @ HFIR, HFIR.T @ zFIR, rcond=None)[0]
                    estKF['P'] = 2000 * np.linalg.inv(HFIR.T @ HFIR)

                    estKF['x'][2:4] = [0, 0]

                    print('Initializing Kalman Filter')

                    h_estKF_trajectory, = ax.plot([], [], '-b')
                    estKF['trajectory'][:, 0] = estKF['x'][:2]
                    estKF['k'] = 0
                else:
                    estKF['initX'].insert(0, estLS['x'])
                    estKF['initk'].insert(0, k)
            else:
                # Update KF using UKF filtering step
                id1 = int(sorted_ids[0])
                id2 = int(sorted_ids[1])
                x1 = sensors['pos'][:, id1]
                x2 = sensors['pos'][:, id2]

                # Measurement
                z12 = -(sensors['peaks'][id1] - sensors['peaks'][id2])

                def r1(x):
                    return np.linalg.norm(x - x1[:, np.newaxis], axis=0)

                def r2(x):
                    return np.linalg.norm(x - x2[:, np.newaxis], axis=0)

                def h(x):
                    r1x = r1(x)
                    r2x = r2(x)
                    # Ensure no division by zero
                    denominator = 2 * r1x * r2x
                    denominator[denominator == 0] = 1e-10
                    cos_value = (r1x ** 2 + r2x ** 2 - np.linalg.norm(x1 - x2) ** 2) / denominator
                    cos_value = np.clip(cos_value, -1.0, 1.0)
                    return np.arccos(cos_value) / ship['radar']['vel']

                # Compute correlations
                elapsed_time_steps = int(sensors['time_step_of_measurement'][id1] - sensors['time_step_of_measurement'][id2])
                if elapsed_time_steps < 0:
                    elapsed_time_steps = 0
                meascorr = np.linalg.matrix_power(estKF['A'], elapsed_time_steps) @ estKF['correlation']
                joint_cov = np.block([
                    [estKF['P'], meascorr.reshape(-1, 1)],
                    [meascorr.reshape(1, -1), 2 * estKF['R']]
                ])

                s, w = sigma_points(np.concatenate((estKF['x'], [0])), joint_cov)

                sz = h(s[0:2, :]) + s[len(estKF['x']):, :]

                zpred = np.dot(sz, w)

                sqxx = (s[0:len(estKF['x']), :] - estKF['x'].reshape(-1, 1)) * np.sqrt(w)
                sqzz = (sz - zpred) * np.sqrt(w)
                Czz = sqzz @ sqzz.T
                Cxz = sqxx @ sqzz.T

                Kk = Cxz @ np.linalg.inv(Czz)
                estKF['x'] = estKF['x'] + Kk @ (z12 - zpred)
                estKF['P'] = estKF['P'] - Kk @ Czz @ Kk.T

                estKF['k'] += 1
                estKF['trajectory'][:, estKF['k']] = estKF['x'][:2]

                # Store correlation for next update
                estKF['correlation'] = -Kk.flatten() * estKF['R']

                print(f'[KF]  max. standard deviation in position: '
                      f'{np.sqrt(np.max(np.linalg.eigvals(estKF["P"][0:2, 0:2]))):.2f} m')

                # *****for Plotting the graph*************#
                # Configure logging
                logging.basicConfig(filename='ukf_std_dev.log', level=logging.INFO,
                                    format='%(asctime)s:%(levelname)s:%(message)s')

                # Replace the print statement with logging
                # Original line:
                # print(f'[KF]  max. standard deviation in position: {np.sqrt(np.max(np.linalg.eigvals(estKF["P"][0:2, 0:2]))):.2f} m')

                # Updated line:
                std_dev_kf = np.sqrt(np.max(np.linalg.eigvals(estKF["P"][0:2, 0:2])))
                logging.info(f'MaxStdDevUKF:{std_dev_kf:.2f}')
                # *****for Plotting the graph*************#

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

        if 'h_KF' in locals():
            h_KF.remove()
            h_KF_ell.remove()
        if not np.isnan(estKF['x']).any():
            h_KF, = ax.plot(estKF['x'][0], estKF['x'][1], '.', color='#0072BD', markersize=7)
            ell = ellipse(estKF['x'][0:2], estKF['P'][0:2, 0:2], 1, 50)
            h_KF_ell, = ax.plot(ell[0, :], ell[1, :], color='#0072BD', linewidth=1, linestyle='-')

            h_estKF_trajectory.set_data(estKF['trajectory'][0, :estKF['k']+1], estKF['trajectory'][1, :estKF['k']+1])

        plt.pause(0.001)
        fig.canvas.flush_events()  # Force update of the plot window

# Turn off interactive mode and show the final plot
plt.ioff()
plt.show()
