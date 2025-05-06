from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt
import time

class OverlayWindow(QWidget):
    def __init__(self, audio_processor):
        super().__init__()
        self.audio_processor = audio_processor

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        h_layout = QHBoxLayout()

        self.label = QLabel("Ожидание текста...")
        self.label.setStyleSheet("""
            font-size: 20px; 
            color: white; 
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 10px;
        """)
        self.label.setWordWrap(True)
        h_layout.addWidget(self.label, 1)

        self.copy_button = QPushButton("●")
        self.copy_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                color: red;
                background-color: black;
                border-radius: 20px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)
        self.copy_button.setFixedSize(40, 40)
        self.copy_button.setToolTip("Копировать текст за последние 30 секунд")
        self.copy_button.clicked.connect(self.copy_text)
        h_layout.addWidget(self.copy_button, 0)

        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 500, 200)

        self.oldPos = None

        self.audio_processor.signals.text_updated.connect(self.update_text)

    def copy_text(self):
        current_time = time.time()
        full_text = self.audio_processor.get_text_for_period(30)
        formatted_time = time.strftime('%H:%M:%S', time.localtime(current_time))
        print(f"[{formatted_time}] {full_text}")
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)

    def update_text(self, text):
        if text:
            self.label.setText(text)
        else:
            self.label.setText("Ожидание текста...")
        self.adjustSize()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_L and event.modifiers() & Qt.AltModifier:
            self.copy_text()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.oldPos:
            delta = event.globalPos() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = None

    def closeEvent(self, event):
        self.audio_processor.stop()
        event.accept()