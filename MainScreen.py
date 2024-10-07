from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QWidget,
)
from BackgroundWidget import BackgroundWidget


# Main screen class
class MainScreen(BackgroundWidget):
    def __init__(self, parent):
        super().__init__(parent, "2941930-fotor-2024090522829.jpg")  # Replace with your image path

        try:
            # Main vertical layout
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(50, 50, 50, 50)  # Adjust margins as needed
            main_layout.setSpacing(20)  # General spacing between elements

            # === Central Content Layout ===
            # Use a vertical layout to stack logos, title, and buttons
            central_layout = QVBoxLayout()
            central_layout.setSpacing(20)  # Spacing between logo, title, and buttons

            self.setup_logos(central_layout)
            self.setup_title(central_layout)
            self.setup_buttons(central_layout)

            # Add central content layout to main layout with stretching
            main_layout.addStretch(1)  # Top stretch
            main_layout.addLayout(central_layout)
            main_layout.addStretch(1)  # Bottom stretch

            # === Footer Text ===
            self.setup_footer(main_layout)

            # Set the main layout
            self.setLayout(main_layout)

        except Exception as e:
            print(f"Error initializing MainScreen: {e}")

    def add_rounded_corners(self, pixmap, radius):
        """
        Returns a new QPixmap with rounded corners.

        :param pixmap: Original QPixmap.
        :param radius: Radius for the rounded corners.
        :return: QPixmap with rounded corners.
        """
        size = pixmap.size()
        rounded = QPixmap(size)
        rounded.fill(Qt.transparent)

        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, size.width(), size.height(), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        return rounded

    def setup_logos(self, parent_layout):
        """Sets up the logo placeholders with rounded edges."""
        logo_layout = QHBoxLayout()
        logo_layout.setSpacing(40)  # Space between logos

        # Left Logo Placeholder
        left_logo = QLabel()
        left_logo.setScaledContents(True)
        left_pixmap = QPixmap("AMS_SVG.png")  # Replace with your left logo path
        if not left_pixmap.isNull():
            # Optionally scale the pixmap to a maximum size
            left_pixmap = left_pixmap.scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # Apply rounded corners
            left_pixmap = self.add_rounded_corners(left_pixmap, 5)
            left_logo.setPixmap(left_pixmap)
        else:
            print("Error: Left logo image not found.")
        left_logo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_logo.setStyleSheet("background-color: transparent;")  # Optional: Transparent background
        logo_layout.addWidget(left_logo, alignment=Qt.AlignRight)

        # Right Logo Placeholder
        right_logo = QLabel()
        right_logo.setScaledContents(True)
        right_pixmap = QPixmap("Signet_FIN_1.jpeg")  # Replace with your right logo path
        if not right_pixmap.isNull():
            # Optionally scale the pixmap to a maximum size
            right_pixmap = right_pixmap.scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            # Apply rounded corners
            right_pixmap = self.add_rounded_corners(right_pixmap, 5)
            right_logo.setPixmap(right_pixmap)
        else:
            print("Error: Right logo image not found.")
        right_logo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_logo.setStyleSheet("background-color: transparent;")  # Optional: Transparent background
        logo_layout.addWidget(right_logo, alignment=Qt.AlignLeft)

        parent_layout.addLayout(logo_layout)

    def setup_title(self, parent_layout):
        """Sets up the title label."""
        title_label = QLabel("Localization of Ship")
        title_label.setFont(QFont("Roboto", 36, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: black;")  # Optional: Set text color
        parent_layout.addWidget(title_label)

    def setup_buttons(self, parent_layout):
        """Sets up the menu buttons."""
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)  # Space between buttons

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
                background-color:#fc9a32;
                border: 1px solid #000000;
            }
        """

        # Start Button
        start_button = QPushButton("Start")
        start_button.setFixedSize(200, 50)
        start_button.setFont(QFont("Roboto", 14, QFont.Bold))
        start_button.setStyleSheet(button_style)
        start_button.setToolTip("Start the Localization Process")
        start_button.setShortcut("Ctrl+S")
        start_button.clicked.connect(self.open_route_assignment)
        button_layout.addWidget(start_button, alignment=Qt.AlignCenter)

        # Credits Button
        credits_button = QPushButton("Credits")
        credits_button.setFixedSize(200, 50)
        credits_button.setFont(QFont("Roboto", 14, QFont.Bold))
        credits_button.setStyleSheet(button_style)
        credits_button.setToolTip("View Credits")
        credits_button.setShortcut("Ctrl+C")
        credits_button.clicked.connect(self.open_credits_screen)
        button_layout.addWidget(credits_button, alignment=Qt.AlignCenter)

        # Exit Button
        exit_button = QPushButton("Exit")
        exit_button.setFixedSize(200, 50)
        exit_button.setFont(QFont("Roboto", 14, QFont.Bold))
        exit_button.setStyleSheet(button_style)
        exit_button.setToolTip("Exit the Application")
        exit_button.setShortcut("Ctrl+E")
        exit_button.clicked.connect(self.exit_application)
        button_layout.addWidget(exit_button, alignment=Qt.AlignCenter)

        parent_layout.addLayout(button_layout)

    def setup_footer(self, parent_layout):
        """Sets up the footer text."""
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)

        footer_label = QLabel(
            "AMS Project: Localization of ship using Particle Filter"
        )
        footer_label.setFont(QFont("Roboto", 12))
        footer_label.setStyleSheet("color: black;")  # Optional: Set text color
        footer_layout.addWidget(
            footer_label, alignment=Qt.AlignLeft | Qt.AlignBottom
        )

        # Add footer to main_layout
        parent_layout.addLayout(footer_layout)

    def open_route_assignment(self):
        try:
            # Switch to route assignment screen
            self.parent().setCurrentWidget(
                self.parent().parent().route_assignment_screen
            )
        except Exception as e:
            print(f"Error switching to route assignment: {e}")

    def open_credits_screen(self):
        try:
            # Switch to credits screen
            self.parent().setCurrentWidget(self.parent().parent().credits_screen)
        except Exception as e:
            print(f"Error switching to credits screen: {e}")

    def exit_application(self):
        try:
            # Close the application
            self.parent().parent().close()
        except Exception as e:
            print(f"Error exiting application: {e}")
