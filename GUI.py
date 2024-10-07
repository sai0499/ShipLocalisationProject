import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from CreditScreen import *
from MainScreen import *
from MapScreen import *
from RouteAssignmentScreen import *

# Main Window with QStackedWidget to hold all screens
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setMinimumSize(1024, 720)

        try:
            # Create QStackedWidget to manage screens
            self.stacked_widget = QStackedWidget()
            self.setCentralWidget(self.stacked_widget)

            # Create screens and add them to the stacked widget
            self.main_screen = MainScreen(self)
            self.route_assignment_screen = RouteAssignmentScreen(self)
            self.map_screen = MapScreen(self)
            self.credits_screen = CreditsScreen(self)

            self.stacked_widget.addWidget(self.main_screen)
            self.stacked_widget.addWidget(self.route_assignment_screen)
            self.stacked_widget.addWidget(self.map_screen)
            self.stacked_widget.addWidget(self.credits_screen)

            # Set the initial screen
            self.stacked_widget.setCurrentWidget(self.main_screen)
        except Exception as e:
            print(f"Error during initialization: {e}")
            self.close()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {e}")
