import os
import sys
import json
import threading
import queue
import time
import numpy as np
import soundcard as sc
import vosk
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import warnings
from soundcard.mediafoundation import SoundcardRuntimeWarning

# Отключаем предупреждения SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
# Укажи путь к папке platforms
plugins_path = r'C:\Users\Пользователь\PycharmProjects\help tech sob\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path

# Путь к модели Vosk
MODEL_PATH = "C:/model/vosk-model-small-ru-0.22"  # Укажи путь к своей модели

# Настройки аудио
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио для более частого обновления


# Класс для сигналов между потоками
class Signals(QObject):
    text_updated = pyqtSignal(str)


# Класс для хранения детальной информации о распознанном тексте
class RecognizedSegment:
    def __init__(self, text, start_time, end_time=None):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time if end_time else start_time

    def __str__(self):
        return self.text


# Класс для записи и распознавания звука
class AudioProcessor(threading.Thread):
    def __init__(self, model_path):
        super().__init__()
        self.daemon = True
        self.audio_queue = queue.Queue()
        self.signals = Signals()
        self.running = True
        self.model_path = model_path
        self.segments = []  # Список сегментов распознанного текста
        self.last_update_time = time.time()
        self.partial_text = ""  # Для хранения частичных результатов

    def run(self):
        # Загружаем модель Vosk
        model = vosk.Model(self.model_path)
        rec = vosk.KaldiRecognizer(model, RATE)
        rec.SetWords(True)  # Включаем информацию о словах и временных метках

        # Запускаем поток для записи звука
        audio_thread = threading.Thread(target=self.record_audio)
        audio_thread.daemon = True
        audio_thread.start()

        # Обрабатываем аудио и распознаем речь
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)

                # Получаем частичные результаты для более плавного обновления
                if rec.PartialResult():
                    partial_result = json.loads(rec.PartialResult())
                    self.partial_text = partial_result.get("partial", "")

                    # Обновляем отображаемый текст с частичными результатами
                    current_time = time.time()
                    if current_time - self.last_update_time > 0.3:  # Обновляем не чаще чем раз в 0.3 секунды
                        display_text = self.get_display_text()
                        self.signals.text_updated.emit(display_text)
                        self.last_update_time = current_time

                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")

                    if text:
                        current_time = time.time()

                        # Если есть информация о словах с временными метками
                        if "result" in result:
                            words = result["result"]
                            for word_info in words:
                                word = word_info["word"]
                                start = word_info.get("start", current_time)
                                end = word_info.get("end", current_time)

                                # Создаем сегмент для каждого слова
                                segment = RecognizedSegment(word, start, end)
                                self.segments.append(segment)
                        else:
                            # Если нет детальной информации о словах, добавляем весь текст как один сегмент
                            segment = RecognizedSegment(text, current_time)
                            self.segments.append(segment)

                        # Очищаем частичные результаты
                        self.partial_text = ""

                        # Удаляем старые сегменты (старше 60 секунд)
                        cutoff_time = current_time - 60
                        self.segments = [s for s in self.segments if s.start_time > cutoff_time]

                        # Обновляем отображаемый текст
                        display_text = self.get_display_text()
                        self.signals.text_updated.emit(display_text)
                        self.last_update_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка распознавания: {e}")
                import traceback
                traceback.print_exc()

    def get_display_text(self):
        """Получает текст для отображения (последние 5 секунд)"""
        current_time = time.time()
        cutoff_time = current_time - 5

        # Собираем тексты сегментов за последние 5 секунд
        recent_texts = [s.text for s in self.segments if s.start_time > cutoff_time]

        # Добавляем текущий частичный результат, если он есть
        display_text = " ".join(recent_texts)
        if self.partial_text:
            if display_text:
                display_text += " " + self.partial_text
            else:
                display_text = self.partial_text

        return display_text

    def get_text_for_period(self, seconds):
        """Получает текст за указанный период времени"""
        current_time = time.time()
        cutoff_time = current_time - seconds

        # Собираем тексты сегментов за указанный период
        period_texts = [s.text for s in self.segments if s.start_time > cutoff_time]

        # Добавляем текущий частичный результат для полноты
        full_text = " ".join(period_texts)
        if self.partial_text and not full_text.endswith(self.partial_text):
            full_text += " " + self.partial_text

        return full_text

    def record_audio(self):
        try:
            # Получаем устройство вывода (колонки)
            speaker = sc.default_speaker()
            # Получаем loopback устройство для захвата звука с колонок
            loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)

            # Получаем обычный микрофон
            default_mic = sc.default_microphone()

            print("Запись звука началась...")

            # Запускаем два рекордера: один для системного звука, другой для микрофона
            with loopback_mic.recorder(samplerate=RATE, channels=CHANNELS) as speaker_rec, \
                    default_mic.recorder(samplerate=RATE, channels=CHANNELS) as mic_rec:
                while self.running:
                    # Записываем фрагменты аудио с обоих источников
                    speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                    mic_data = mic_rec.record(numframes=CHUNK_SIZE)

                    # Смешиваем аудио из обоих источников
                    mixed_data = np.mean([speaker_data, mic_data], axis=0)

                    # Преобразуем в 16-битный формат
                    audio_data = (mixed_data * 32767).astype(np.int16).tobytes()

                    # Отправляем в очередь для распознавания
                    self.audio_queue.put(audio_data)

        except Exception as e:
            print(f"Ошибка записи звука: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False


# Класс оверлея
class OverlayWindow(QWidget):
    def __init__(self, audio_processor):
        super().__init__()
        self.audio_processor = audio_processor

        # Настройка окна как оверлея
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Создаем главный вертикальный layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Создаем горизонтальный layout для текста и кнопки
        h_layout = QHBoxLayout()

        # Создаем метку с текстом
        self.label = QLabel("Ожидание текста...")
        self.label.setStyleSheet("""
            font-size: 20px; 
            color: white; 
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 10px;
        """)
        self.label.setWordWrap(True)  # Разрешаем перенос строк
        h_layout.addWidget(self.label, 1)  # 1 - это вес (растягивание)

        # Создаем кнопку копирования
        self.copy_button = QPushButton("Копировать")
        self.copy_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 180, 200);
            border-radius: 10px;
            padding: 10px;
            margin-left: 5px;
        """)
        self.copy_button.setFixedWidth(120)  # Фиксированная ширина кнопки
        self.copy_button.clicked.connect(self.copy_text)
        h_layout.addWidget(self.copy_button, 0)  # 0 - это вес (не растягивать)

        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 500, 200)  # Увеличиваем размер окна

        # Для перетаскивания окна
        self.oldPos = None

        # Подключаем сигнал обновления текста
        self.audio_processor.signals.text_updated.connect(self.update_text)

    def copy_text(self):
        # Функция копирования текста (та же, что и при нажатии Alt+L)
        full_text = self.audio_processor.get_text_for_period(30)
        print("\n=== Текст за последние 30 секунд ===")
        print(full_text)
        print("===================================\n")

        # Копируем текст в буфер обмена
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)
        print("Текст скопирован в буфер обмена")

    def update_text(self, text):
        if text:
            self.label.setText(text)
        else:
            self.label.setText("Ожидание текста...")
        # Изменяем размер окна в зависимости от текста
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
        # Останавливаем аудио процессор при закрытии окна
        self.audio_processor.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Создаем и запускаем аудио процессор
    audio_processor = AudioProcessor(MODEL_PATH)
    audio_processor.start()

    # Создаем оверлей
    overlay = OverlayWindow(audio_processor)
    overlay.show()

    sys.exit(app.exec_())