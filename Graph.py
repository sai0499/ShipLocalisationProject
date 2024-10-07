from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
import numpy as np
import os
import time
import threading
from collections import deque

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create a figure to plot
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        # Initialize the canvas where the graph will be drawn
        super(GraphCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # Set labels, limits, and title
        self.ax.set_xlim(0, 300)  # Assuming 300 seconds simulation
        self.ax.set_ylim(0, 50)   # Adjust based on expected standard deviations
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Standard Deviation (m)')
        self.ax.set_title('Comparing Standard Deviations of Particle Filter and Unscented Kalman Filter')

        # Initialize data containers
        self.time_pf = deque(maxlen=3000)  # Store last 3000 points (~300 seconds at 10 Hz)
        self.std_pf = deque(maxlen=3000)

        self.time_ukf = deque(maxlen=3000)
        self.std_ukf = deque(maxlen=3000)

        self.reference = deque(maxlen=3000)  # Placeholder for comparison reference

        # Initialize plot lines
        self.line_pf, = self.ax.plot([], [], label='PF Std Dev', color='red')
        self.line_ukf, = self.ax.plot([], [], label='UKF Std Dev', color='blue')
        self.line_ref, = self.ax.plot([], [], label='Reference', color='green', linestyle='--')

        self.ax.legend(loc='upper right')

        # Start monitoring log files
        self.pf_log = 'pf_std_dev.log'
        self.ukf_log = 'ukf_std_dev.log'

        # Initialize file positions
        self.pf_pos = 0
        self.ukf_pos = 0

        # Start threads to monitor log files
        self.running = True
        self.thread_pf = threading.Thread(target=self.monitor_log, args=(self.pf_log, 'PF'))
        self.thread_ukf = threading.Thread(target=self.monitor_log, args=(self.ukf_log, 'UKF'))
        self.thread_pf.start()
        self.thread_ukf.start()

        # Start animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)

    def monitor_log(self, logfile, script_type):
        """Monitor the given log file for new standard deviation entries."""
        while self.running:
            if os.path.exists(logfile):
                with open(logfile, 'r') as f:
                    # Move to the last read position
                    f.seek(self.pf_pos if script_type == 'PF' else self.ukf_pos)
                    lines = f.readlines()
                    if lines:
                        for line in lines:
                            if script_type == 'PF' and 'MaxStdDevPF:' in line:
                                try:
                                    std = float(line.strip().split('MaxStdDevPF:')[1])
                                    current_time = time.time() - self.start_time
                                    self.time_pf.append(current_time)
                                    self.std_pf.append(std)
                                except ValueError:
                                    pass  # Handle parsing error
                            elif script_type == 'UKF' and 'MaxStdDevUKF:' in line:
                                try:
                                    std = float(line.strip().split('MaxStdDevUKF:')[1])
                                    current_time = time.time() - self.start_time
                                    self.time_ukf.append(current_time)
                                    self.std_ukf.append(std)
                                except ValueError:
                                    pass  # Handle parsing error
                    # Update file position
                    if script_type == 'PF':
                        self.pf_pos = f.tell()
                    else:
                        self.ukf_pos = f.tell()
            time.sleep(0.1)  # Poll every 100 ms

    def update_plot(self, frame):
        """Update the plot with new data."""
        # Update PF line
        if self.time_pf and self.std_pf:
            self.line_pf.set_data(self.time_pf, self.std_pf)

        # Update UKF line
        if self.time_ukf and self.std_ukf:
            self.line_ukf.set_data(self.time_ukf, self.std_ukf)

        # Update Reference line (for example, a fixed value or another data source)
        # Here, we'll use a dummy reference of 25 m
        current_time = max(
            self.time_pf[-1] if self.time_pf else 0,
            self.time_ukf[-1] if self.time_ukf else 0
        )
        self.reference.append(25)  # Example reference value
        self.line_ref.set_data(range(len(self.reference)), self.reference)

        # Adjust x-axis based on current_time
        self.ax.set_xlim(max(0, current_time - 300), current_time + 10)  # Show last 300 seconds

        # Adjust y-axis if necessary
        all_std = list(self.std_pf) + list(self.std_ukf) + list(self.reference)
        if all_std:
            min_std = min(all_std) - 5
            max_std = max(all_std) + 5
            self.ax.set_ylim(max(0, min_std), max_std)

        self.fig.canvas.draw()

    def closeEvent(self, event):
        """Handle the closing of the GraphCanvas."""
        self.running = False
        self.thread_pf.join()
        self.thread_ukf.join()
        event.accept()
