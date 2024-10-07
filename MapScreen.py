from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QFrame,
)

from BackgroundWidget import BackgroundWidget
from Graph import GraphCanvas
from MatlabWorker import MATLABWorker

image_Path = "2941930-fotor-2024090522829.jpg"  # Background Image path


# Map visualization screen class
class MapScreen(BackgroundWidget):
    def __init__(self, parent):
        super().__init__(parent, image_Path)  # Use the same background

        try:
            layout = QVBoxLayout()

            # Add the title "Performance Graph" at the top of the screen
            title_label = QLabel("Performance Graph")
            title_label.setFont(QFont("Roboto", 24, QFont.Bold))
            title_label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(title_label)

            # Create a frame to hold the graph canvas with rounded edges
            canvas_frame = QFrame()
            canvas_frame.setStyleSheet(
                """
                QFrame {
                    border: 2px solid #000000;
                    border-radius: 15px;
                    background-color: #ffffff;
                }
                """
            )
            canvas_layout = QVBoxLayout()
            canvas_layout.setContentsMargins(10, 10, 10, 10)  # Optional padding
            self.graph_canvas = GraphCanvas(self, width=5, height=10)
            canvas_layout.addWidget(self.graph_canvas)
            canvas_frame.setLayout(canvas_layout)
            layout.addWidget(canvas_frame)

            # Button layout
            button_layout = QHBoxLayout()

            # Common stylesheet for all buttons
            button_style = """
                QPushButton {
                    background-color: #eabf7d;
                    color: white;
                    border: 1px solid #000000;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #f5d0a9;
                    border: 1px solid #000000;
                }
                QPushButton:pressed {
                    background-color: #fc9a32;
                    border: 1px solid #000000;
                }
            """

            # Particle Filter button
            pf_button = QPushButton("Particle Filter")
            pf_button.setFixedSize(200, 50)
            pf_button.setFont(QFont("Roboto", 14, QFont.Bold))
            pf_button.setStyleSheet(button_style)
            pf_button.clicked.connect(self.run_particle_filter)
            button_layout.addWidget(pf_button)

            # UKFilter button
            ukf_button = QPushButton("UKFilter")
            ukf_button.setFixedSize(200, 50)
            ukf_button.setFont(QFont("Roboto", 14, QFont.Bold))
            ukf_button.setStyleSheet(button_style)
            ukf_button.clicked.connect(self.run_ukfilter)
            button_layout.addWidget(ukf_button)

            # Graph button
            graph_button = QPushButton("Graph")
            graph_button.setFixedSize(200, 50)
            graph_button.setFont(QFont("Roboto", 14, QFont.Bold))
            graph_button.setStyleSheet(button_style)
            graph_button.clicked.connect(self.visualize_graph)
            button_layout.addWidget(graph_button)

            # Back button
            back_button = QPushButton("Back")
            back_button.setFixedSize(200, 50)
            back_button.setFont(QFont("Roboto", 14, QFont.Bold))
            back_button.setStyleSheet(button_style)
            back_button.clicked.connect(self.back_to_route_assignment)
            button_layout.addWidget(back_button)

            layout.addLayout(button_layout)
            self.setLayout(layout)

        except Exception as e:
            print(f"Error initializing MapScreen: {e}")

    def visualize_graph(self):
        # Call the plot_graph method to draw the graph
        self.graph_canvas.plot_graph()

    def run_particle_filter(self):
        # Run the Particle Filter in a separate thread
        self.worker = MATLABWorker('particle_filter')
        self.worker.finished.connect(self.on_matlab_finished)
        self.worker.error.connect(self.on_matlab_error)
        self.worker.start()  # Start the worker thread

    def run_ukfilter(self):
        # Run the UKFilter in a separate thread
        self.worker = MATLABWorker('ukf')
        self.worker.finished.connect(self.on_matlab_finished)
        self.worker.error.connect(self.on_matlab_error)
        self.worker.start()  # Start the worker thread

    def on_matlab_finished(self):
        # Handle actions when MATLAB processing is done
        print("MATLAB processing finished successfully.")

    def on_matlab_error(self, error_message):
        # Handle errors from MATLAB
        QMessageBox.critical(self, "MATLAB Error", f"An error occurred: {error_message}")

    def back_to_route_assignment(self):
        try:
            self.parent().setCurrentWidget(self.parent().parent().route_assignment_screen)
        except Exception as e:
            print(f"Error going back to route assignment screen: {e}")
