from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QWidget

# Base widget to handle background painting
class BackgroundWidget(QWidget):
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.image_path = image_path

    def paintEvent(self, event):
        try:
            if self.image_path:
                painter = QPainter(self)
                pixmap = QPixmap(self.image_path)
                painter.drawPixmap(self.rect(), pixmap)
        except Exception as e:
            print(f"Error while painting background: {e}")
