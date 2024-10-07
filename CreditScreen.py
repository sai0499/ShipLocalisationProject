from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QPushButton

from BackgroundWidget import BackgroundWidget

image_Path = "2941930-fotor-2024090522829.jpg"  # Background Image path

#Credits Screen for displaying all the credits and references
class CreditsScreen(BackgroundWidget):
    def __init__(self, parent):
        super().__init__(parent, image_Path)  # Use the same background

        try:
            layout = QVBoxLayout()

            # Title Label
            title_label = QLabel("Credits")
            title_font = QFont("Roboto", 24, QFont.Bold)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)

            # Grid Layout for Credits
            grid_layout = QGridLayout()

            # Define a font and color for the labels
            label_font = QFont("Roboto", 14)
            value_font = QFont("Roboto", 14)
            value_color = QColor(0, 0, 0)

            credits = [
                ("Project Name", "Ship Localization"),
                ("", ""),
                ("Professor", "Prof. Dr.-Ing. Benjamin Noack"),
                ("", ""),
                ("Group Member #1", "Sreerama Sai Vyshnavi"),
                ("Group Member #2", "Sahithya Patel"),
                ("Group Member #3", "Sunita Nalatwad"),
                ("Group Member #4", "Sai Teja Dampanaboina"),
                ("", ""),
                ("UI Background Image", "Designed by pikisuperstar / Freepik"),
            ]

            for row, (role, person) in enumerate(credits):
                label = QLabel(role)
                label.setFont(label_font)
                value = QLabel(person)
                value.setFont(value_font)
                value.setStyleSheet(f"color: rgb({value_color.red()}, {value_color.green()}, {value_color.blue()});")
                grid_layout.addWidget(label, row, 0, alignment=Qt.AlignHCenter)
                grid_layout.addWidget(value, row, 1, alignment=Qt.AlignHCenter)

            layout.addLayout(grid_layout)

            # Create Back button
            back_button = QPushButton("Back")
            back_button.setFixedSize(120, 50)
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
            layout.addWidget(back_button)

            layout.setAlignment(back_button, Qt.AlignHCenter | Qt.AlignBottom)
            self.setLayout(layout)
        except Exception as e:
            print(f"Error initializing CreditsScreen: {e}")

    def back_to_main_screen(self):
        try:
            self.parent().setCurrentWidget(self.parent().parent().main_screen)
        except Exception as e:
            print(f"Error going back to main screen: {e}")
