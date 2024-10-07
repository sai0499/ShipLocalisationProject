from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFileDialog, QMessageBox

from BackgroundWidget import BackgroundWidget

image_Path = "2941930-fotor-2024090522829.jpg"  # Background Image path
# Route assignment screen class
class RouteAssignmentScreen(BackgroundWidget):
    def __init__(self, parent):
        super().__init__(parent, image_Path)  # Use the same background

        try:
            layout = QVBoxLayout()

            # Add the title "Route Assignment" at the top of the screen
            title_label = QLabel("Ship Course Assignment")
            title_label.setFont(QFont("Roboto", 24, QFont.Bold))

            title_label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(title_label)

            # Center the CSV selection section
            center_layout = QVBoxLayout()
            center_layout.setAlignment(Qt.AlignCenter)

            # Text above the white box
            instruction_label = QLabel(
                "Please select a CSV file with Latitude and Longitude Coordinates to assign a course to the Ship")
            instruction_label.setFont(QFont("Roboto", 10))  # Set font size to 10
            instruction_label.setAlignment(Qt.AlignHCenter)
            center_layout.addWidget(instruction_label)  # Add instruction above the white box

            # Horizontal layout for CSV selection
            file_layout = QHBoxLayout()

            # White box to display selected file
            self.selected_file_label = QLabel("")
            self.selected_file_label.setFixedSize(500, 30)
            self.selected_file_label.setStyleSheet(
                "background-color: white; border: 1px solid #000000; border-radius: 10px;")
            self.selected_file_label.setAlignment(Qt.AlignCenter)
            file_layout.addWidget(self.selected_file_label)

            # Create select file button
            select_file_button = QPushButton("Select CSV File")
            select_file_button.setFixedSize(200, 50)
            select_file_button.setFont(QFont("Roboto", 14, QFont.Bold))
            select_file_button.setStyleSheet(
                """QPushButton {background-color: #eabf7d;
                               color: white;
                               border: 1px solid #000000;
                               border-radius: 10px;
                           }
                  QPushButton:hover {background-color: #f5d0a9;
                                     border: 1px solid #000000;
                           }
                  QPushButton:pressed {background-color:#fc9a32;
                                       border: 1px solid #000000;
                           }
               """)
            select_file_button.clicked.connect(self.open_file_dialog)
            file_layout.addWidget(select_file_button)

            layout.addStretch()
            center_layout.addLayout(file_layout)
            layout.addLayout(center_layout)

            # Horizontal layout for Back and Visualize buttons
            button_layout = QHBoxLayout()

            # Create Back button aligned to the left
            back_button = QPushButton("Back")
            back_button.setFixedSize(150, 50)
            back_button.setFont(QFont("Roboto", 14, QFont.Bold))
            back_button.setStyleSheet(
                """QPushButton {background-color: #eabf7d;
                               color: white;
                               border: 1px solid #000000;
                               border-radius: 10px;
                           }
                  QPushButton:hover {background-color: #f5d0a9;
                                     border: 1px solid #000000;
                           }
                  QPushButton:pressed {background-color:#fc9a32;
                                       border: 1px solid #000000;
                           }
               """)
            back_button.clicked.connect(self.back_to_main_screen)
            button_layout.addWidget(back_button)

            # Add stretch to push Visualize button to the right
            button_layout.addStretch()

            # Create Visualize button aligned to the right
            self.visualize_button = QPushButton("Assign")
            self.visualize_button.setFixedSize(150, 50)
            self.visualize_button.setFont(QFont("Roboto", 14, QFont.Bold))
            self.visualize_button.setStyleSheet(
                """QPushButton {background-color: #eabf7d;
                               color: white;
                               border: 1px solid #000000;
                               border-radius: 10px;
                           }
                  QPushButton:hover {background-color: #f5d0a9;
                                     border: 1px solid #000000;
                           }
                  QPushButton:pressed {background-color:#fc9a32;
                                       border: 1px solid #000000;
                           }
               """)
            # self.visualize_button.setEnabled(False)  # Disable button initially
            self.visualize_button.clicked.connect(self.visualize_map_screen)
            button_layout.addWidget(self.visualize_button)

            # Align the button layout at the bottom
            layout.addStretch()  # Push buttons to the bottom
            layout.addLayout(button_layout)

            self.setLayout(layout)

            self.selected_file_path = None  # To store the file path

        except Exception as e:
            print(f"Error initializing RouteAssignmentScreen: {e}")

    def open_file_dialog(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
            if file_name:
                self.selected_file_label.setText(file_name)
                self.selected_file_path = file_name
            # self.visualize_button.setEnabled(True)  # Enable the button when a file is selected
        except Exception as e:
            print(f"Error selecting file: {e}")

    def visualize_map_screen(self):
        try:
            if self.selected_file_path:  # Ensure a file is selected (if required)
                # Switch to map screen
                self.parent().setCurrentWidget(self.parent().parent().map_screen)

                # Call the method to plot the graph in the MapScreen
                self.parent().parent().map_screen.visualize_graph()

            else:
                QMessageBox.warning(self, "No File Selected", "Please select a CSV file before proceeding.")
        except Exception as e:
            print(f"Error visualizing map screen: {e}")

    def back_to_main_screen(self):
        try:
            self.selected_file_label.clear()
            self.selected_file_path = None
            self.parent().setCurrentWidget(self.parent().parent().main_screen)
        except Exception as e:
            print(f"Error going back to main screen: {e}")

