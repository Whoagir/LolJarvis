import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_num_threads(1)

from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QPushButton,
                             QHBoxLayout, QApplication, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import time
from audio_recorder import AudioRecorder
import multiprocessing as mp
import faulthandler, sys, traceback
faulthandler.enable()
os.environ["PYTHONFAULTHANDLER"] = "1"

class RequestProcess(mp.Process):
    def __init__(self, conn, query, model="gpt-4o-mini"):
        super().__init__()
        self.conn  = conn
        self.query = query
        self.model = model

    def run(self):
        try:
            print("[RequestProcess] start, pid =", os.getpid(), flush=True)
            from duckai import DuckAI
            duck = DuckAI()
            self.conn.send("[RequestProcess] Запрос отправлен…")
            resp = duck.chat(self.query, model=self.model)
            self.conn.send(resp)
        except Exception as e:
            tb = traceback.format_exc()
            self.conn.send(f"[RequestProcess] Python-ошибка: {e}\n{tb}")
        finally:
            self.conn.close()

class TranscriptionWindow(QWidget):
    def __init__(self, audio_recorder):
        super().__init__()
        self.audio_recorder = audio_recorder
        self.is_recording = False
        self.is_paused = False

        # Настройка окна
        self.setWindowTitle("Транскрибация аудио")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #2D2D30; color: white;")

        # Создание макета
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Область вывода текста
        self.text_output = QLabel("Здесь будет отображаться транскрибированный текст")
        self.text_output.setStyleSheet("""
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 15px;
            font-size: 16px;
            color: #FFFFFF;
        """)
        self.text_output.setWordWrap(True)
        self.text_output.setMinimumHeight(200)
        main_layout.addWidget(self.text_output)

        # Индикатор прогресса для транскрибации
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #333333;
                border-radius: 5px;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)

        # Статус
        self.status_label = QLabel("Готов к записи")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #AAAAAA; font-size: 14px;")
        main_layout.addWidget(self.status_label)

        # Контейнер для кнопок
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        # Кнопка записи
        self.record_button = QPushButton("Начать запись")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F44336;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        buttons_layout.addWidget(self.record_button)

        # Кнопка очистки
        self.clear_button = QPushButton("Очистить запись")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #455A64;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #37474F;
            }
        """)
        self.clear_button.clicked.connect(self.clear_recording)
        buttons_layout.addWidget(self.clear_button)

        # Кнопка транскрибации
        self.transcribe_button = QPushButton("Транскрибировать")
        self.transcribe_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
            QPushButton:pressed {
                background-color: #1E88E5;
            }
            QPushButton:disabled {
                background-color: #757575;
                color: #BDBDBD;
            }
        """)
        self.transcribe_button.clicked.connect(self.transcribe_audio)
        self.transcribe_button.setEnabled(False)
        buttons_layout.addWidget(self.transcribe_button)

        # Кнопка отправки запроса
        self.send_request_button = QPushButton("Отправить запрос")
        self.send_request_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #43A047;
            }
            QPushButton:disabled {
                background-color: #757575;
                color: #BDBDBD;
            }
        """)
        self.send_request_button.clicked.connect(self.send_request)
        self.send_request_button.setEnabled(False)
        buttons_layout.addWidget(self.send_request_button)

        main_layout.addLayout(buttons_layout)

        # Таймер записи
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_recording_time)
        self.recording_start_time = 0
        self.recording_elapsed_time = 0

        # Счетчик времени записи
        self.time_label = QLabel("00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #E53935; font-size: 18px; font-weight: bold;")
        main_layout.addWidget(self.time_label)

        # Добавляем поле для вывода ответа
        self.response_label = QLabel("Ответ:")
        self.response_label.setStyleSheet("color: #AAAAAA; font-size: 14px; margin-top: 10px;")
        main_layout.addWidget(self.response_label)

        self.response_output = QTextEdit()
        self.response_output.setStyleSheet("""
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 15px;
            font-size: 16px;
            color: #FFFFFF;
        """)
        self.response_output.setReadOnly(True)
        self.response_output.setMinimumHeight(150)
        main_layout.addWidget(self.response_output)

        self.setLayout(main_layout)
        self.resize(600, 600)

        # Подключение сигналов от аудио рекордера
        self.audio_recorder.signals.transcription_complete.connect(self.handle_transcription_complete)
        self.audio_recorder.signals.transcription_progress.connect(self.handle_transcription_progress)

    def toggle_recording(self):
        if not self.is_recording:
            # Начать запись
            self.is_recording = True
            self.is_paused = False
            self.audio_recorder.start_recording()
            self.record_button.setText("Пауза")
            self.status_label.setText("Запись...")
            self.status_label.setStyleSheet("color: #E53935; font-size: 14px;")
            self.recording_start_time = time.time() - self.recording_elapsed_time
            self.timer.start(1000)  # Обновление каждую секунду
            self.transcribe_button.setEnabled(False)
            self.send_request_button.setEnabled(False)
        elif not self.is_paused:
            # Пауза записи
            self.is_paused = True
            self.audio_recorder.pause_recording()
            self.record_button.setText("Продолжить")
            self.status_label.setText("Пауза")
            self.status_label.setStyleSheet("color: #FFC107; font-size: 14px;")
            self.timer.stop()
            self.recording_elapsed_time = time.time() - self.recording_start_time
            self.transcribe_button.setEnabled(True)
        else:
            # Продолжить запись
            self.is_paused = False
            self.audio_recorder.resume_recording()
            self.record_button.setText("Пауза")
            self.status_label.setText("Запись...")
            self.status_label.setStyleSheet("color: #E53935; font-size: 14px;")
            self.recording_start_time = time.time() - self.recording_elapsed_time
            self.timer.start(1000)
            self.transcribe_button.setEnabled(False)
            self.send_request_button.setEnabled(False)

    def clear_recording(self):
        self.audio_recorder.clear_recording()
        self.text_output.setText("Здесь будет отображаться транскрибированный текст")
        self.status_label.setText("Запись очищена")
        self.status_label.setStyleSheet("color: #AAAAAA; font-size: 14px;")

        # Сброс таймера
        self.timer.stop()
        self.recording_elapsed_time = 0
        self.time_label.setText("00:00")

        # Если запись была активна, сбрасываем состояние
        if self.is_recording:
            self.is_recording = False
            self.is_paused = False
            self.record_button.setText("Начать запись")

        self.transcribe_button.setEnabled(False)
        self.send_request_button.setEnabled(False)
        self.response_output.clear()

    def transcribe_audio(self):
        if self.audio_recorder.has_recording():
            self.status_label.setText("Транскрибация...")
            self.status_label.setStyleSheet("color: #2196F3; font-size: 14px;")
            self.transcribe_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.send_request_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.show()

            # Запускаем транскрибацию в отдельном потоке
            self.audio_recorder.transcribe()
        else:
            self.status_label.setText("Нет записи для транскрибации")
            self.status_label.setStyleSheet("color: #F44336; font-size: 14px;")

    def update_recording_time(self):
        if not self.is_paused:
            self.recording_elapsed_time = time.time() - self.recording_start_time

        minutes = int(self.recording_elapsed_time // 60)
        seconds = int(self.recording_elapsed_time % 60)
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")

    def handle_transcription_complete(self, text):
        self.text_output.setText(text)
        self.status_label.setText("Транскрибация завершена")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px;")
        self.transcribe_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.send_request_button.setEnabled(True)
        self.progress_bar.hide()

    def handle_transcription_progress(self, progress):
        self.progress_bar.setValue(int(progress))

    def send_request(self):
        transcribed_text = self.text_output.text()
        if not transcribed_text or transcribed_text.startswith("Здесь будет"):
            self._set_status("Нет текста для отправки запроса", bad=True)
            return

        prompt = ("Представь ты на собеседование мидл python разработчик, "
                  "тебе надо коротко ответить на вопрос. "
                  "Учитывай, что я могу опечататься…  Вот вопрос:")
        query = f"{prompt} {transcribed_text}"
        print("[GUI] Отправляемый запрос:", query, flush=True)

        # Создаём канал для общения
        parent_conn, child_conn = mp.Pipe()
        # Запускаем процесс
        self.req_proc = RequestProcess(child_conn, query)
        self.req_proc.start()

        # Опрашиваем канал таймером, чтобы не блокировать GUI
        self.req_timer = QTimer()
        self.req_timer.timeout.connect(lambda: self._poll_reply(parent_conn))
        self.req_timer.start(100)       # 10 раз в секунду
        self._set_status("Отправка запроса…")

    def _poll_reply(self, conn):
        while conn.poll():
            msg = conn.recv()
            if msg.startswith("[RequestProcess]") and "Python-ошибка" not in msg:
                print(msg)
            else:
                self.handle_response(msg)
                self.req_timer.stop()
                if self.req_proc.is_alive():
                    self.req_proc.join(timeout=0.5)
                break

    def handle_response(self, response):
        if "Python-ошибка" in response:
            self.response_output.setText(response)
            self._set_status("Ошибка при получении ответа", bad=True)
        else:
            self.response_output.setText(response)
            self._set_status("Ответ получен")
        self.send_request_button.setEnabled(True)

    def _set_status(self, txt, bad=False):
        color = "#F44336" if bad else "#4CAF50"
        self.status_label.setText(txt)
        self.status_label.setStyleSheet(f"color:{color}; font-size:14px;")

    def closeEvent(self, event):
        self.audio_recorder.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    window = TranscriptionWindow(recorder)
    window.show()
    sys.exit(app.exec_())