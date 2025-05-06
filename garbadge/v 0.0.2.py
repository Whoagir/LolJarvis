import os
import sys
import json
import threading
import queue
import time
import numpy as np
import soundcard as sc
import vosk
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSlider, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
import warnings
from soundcard.mediafoundation import SoundcardRuntimeWarning

plugins_path = r'C:\Users\Пользователь\PycharmProjects\help tech sob\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path

# Отключаем предупреждения SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

# Путь к модели Vosk
MODEL_PATH = "C:/model/vosk-model-small-ru-0.22"

# Настройки аудио
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио


# Класс для сигналов между потоками
class Signals(QObject):
    text_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    audio_level_updated = pyqtSignal(float)


# Класс для хранения распознанного текста
class TextBuffer:
    def __init__(self, max_age_seconds=120):
        self.segments = []
        self.max_age_seconds = max_age_seconds
        self.partial_text = ""
        self.lock = threading.Lock()  # Для потокобезопасности

    def add_segment(self, text, start_time, end_time=None):
        with self.lock:
            self.segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time if end_time else start_time
            })
            self._cleanup_old_segments()

    def set_partial(self, text):
        with self.lock:
            self.partial_text = text

    def get_text(self, seconds=5):
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - seconds

            # Собираем тексты сегментов за указанный период
            recent_texts = [s['text'] for s in self.segments
                            if s['start_time'] > cutoff_time]

            # Объединяем тексты и добавляем частичный результат
            result = " ".join(recent_texts)
            if self.partial_text and not result.endswith(self.partial_text):
                if result:
                    result += " " + self.partial_text
                else:
                    result = self.partial_text

            return result

    def _cleanup_old_segments(self):
        current_time = time.time()
        cutoff_time = current_time - self.max_age_seconds
        self.segments = [s for s in self.segments if s['start_time'] > cutoff_time]


# Класс для записи и обработки аудио
class AudioProcessor(threading.Thread):
    def __init__(self, model_path):
        super().__init__()
        self.daemon = True
        self.audio_queue = queue.Queue()
        self.signals = Signals()
        self.running = True
        self.model_path = model_path
        self.text_buffer = TextBuffer()
        self.last_update_time = time.time()

        # Настройки источников аудио
        self.use_mic = True
        self.use_speaker = True
        self.mic_volume = 1.0
        self.speaker_volume = 1.0

        # Для определения уровня аудио
        self.audio_level = 0.0

        # Доступные устройства
        self.speakers = []
        self.microphones = []
        self.selected_speaker = None
        self.selected_mic = None

        self._init_audio_devices()

    def _init_audio_devices(self):
        try:
            # Получаем список всех устройств
            self.speakers = sc.all_speakers()
            self.microphones = sc.all_microphones()

            # Устанавливаем устройства по умолчанию
            self.selected_speaker = sc.default_speaker()
            self.selected_mic = sc.default_microphone()

            self.signals.status_updated.emit("Аудио устройства инициализированы")
        except Exception as e:
            self.signals.status_updated.emit(f"Ошибка инициализации аудио: {e}")
            print(f"Ошибка инициализации аудио: {e}")

    def set_audio_source(self, use_mic, use_speaker):
        self.use_mic = use_mic
        self.use_speaker = use_speaker

    def set_volumes(self, mic_volume, speaker_volume):
        self.mic_volume = mic_volume
        self.speaker_volume = speaker_volume

    def set_devices(self, speaker_id=None, mic_id=None):
        if speaker_id is not None:
            for speaker in self.speakers:
                if speaker.id == speaker_id:
                    self.selected_speaker = speaker
                    break

        if mic_id is not None:
            for mic in self.microphones:
                if mic.id == mic_id:
                    self.selected_mic = mic
                    break

    def run(self):
        try:
            # Загружаем модель Vosk
            self.signals.status_updated.emit("Загрузка модели распознавания...")
            model = vosk.Model(self.model_path)
            rec = vosk.KaldiRecognizer(model, RATE)
            rec.SetWords(True)  # Включаем информацию о словах и временных метках
            self.signals.status_updated.emit("Модель загружена")

            # Запускаем поток для записи звука
            audio_thread = threading.Thread(target=self.record_audio)
            audio_thread.daemon = True
            audio_thread.start()

            # Обрабатываем аудио и распознаем речь
            while self.running:
                try:
                    audio_data = self.audio_queue.get(timeout=1)

                    # Обновляем уровень аудио (для визуализации)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                    self.audio_level = np.max(np.abs(audio_array))
                    self.signals.audio_level_updated.emit(self.audio_level)

                    # Получаем частичные результаты
                    if rec.PartialResult():
                        partial_result = json.loads(rec.PartialResult())
                        partial_text = partial_result.get("partial", "")
                        self.text_buffer.set_partial(partial_text)

                        # Обновляем UI не чаще чем раз в 0.3 секунды
                        current_time = time.time()
                        if current_time - self.last_update_time > 0.3:
                            display_text = self.text_buffer.get_text(5)
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
                                    self.text_buffer.add_segment(word, start, end)
                            else:
                                # Если нет детальной информации о словах
                                self.text_buffer.add_segment(text, current_time)

                            # Очищаем частичные результаты
                            self.text_buffer.set_partial("")

                            # Обновляем отображаемый текст
                            display_text = self.text_buffer.get_text(5)
                            self.signals.text_updated.emit(display_text)
                            self.last_update_time = current_time

                except queue.Empty:
                    continue
                except Exception as e:
                    self.signals.status_updated.emit(f"Ошибка распознавания: {e}")
                    print(f"Ошибка распознавания: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)  # Пауза перед повторной попыткой

        except Exception as e:
            self.signals.status_updated.emit(f"Критическая ошибка: {e}")
            print(f"Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()

    def record_audio(self):
        try:
            speaker_rec = None
            mic_rec = None

            while self.running:
                try:
                    # Проверяем и обновляем рекордеры если нужно
                    if self.use_speaker and not speaker_rec and self.selected_speaker:
                        loopback_mic = sc.get_microphone(self.selected_speaker.id, include_loopback=True)
                        speaker_rec = loopback_mic.recorder(samplerate=RATE, channels=CHANNELS)
                        self.signals.status_updated.emit("Захват системного звука активирован")

                    if self.use_mic and not mic_rec and self.selected_mic:
                        mic_rec = self.selected_mic.recorder(samplerate=RATE, channels=CHANNELS)
                        self.signals.status_updated.emit("Захват микрофона активирован")

                    # Записываем фрагменты аудио с активных источников
                    speaker_data = None
                    mic_data = None

                    if speaker_rec and self.use_speaker:
                        speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                        speaker_data = speaker_data * self.speaker_volume

                    if mic_rec and self.use_mic:
                        mic_data = mic_rec.record(numframes=CHUNK_SIZE)
                        mic_data = mic_data * self.mic_volume

                    # Смешиваем аудио из активных источников
                    if speaker_data is not None and mic_data is not None:
                        # Интеллектуальное смешивание: выбираем источник с большей амплитудой
                        speaker_max = np.max(np.abs(speaker_data))
                        mic_max = np.max(np.abs(mic_data))

                        if speaker_max > mic_max * 1.5:  # Если системный звук значительно громче
                            mixed_data = speaker_data
                        elif mic_max > speaker_max * 1.5:  # Если микрофон значительно громче
                            mixed_data = mic_data
                        else:  # Иначе смешиваем с весами
                            mixed_data = speaker_data * 0.6 + mic_data * 0.4
                    elif speaker_data is not None:
                        mixed_data = speaker_data
                    elif mic_data is not None:
                        mixed_data = mic_data
                    else:
                        # Если нет активных источников, создаем тишину
                        mixed_data = np.zeros((CHUNK_SIZE, CHANNELS))

                    # Преобразуем в 16-битный формат
                    audio_data = (mixed_data * 32767).astype(np.int16).tobytes()

                    # Отправляем в очередь для распознавания
                    self.audio_queue.put(audio_data)

                except Exception as e:
                    self.signals.status_updated.emit(f"Ошибка записи: {e}")
                    print(f"Ошибка записи: {e}")

                    # Закрываем рекордеры при ошибке
                    if speaker_rec:
                        speaker_rec.close()
                        speaker_rec = None
                    if mic_rec:
                        mic_rec.close()
                        mic_rec = None

                    time.sleep(1)  # Пауза перед повторной попыткой

            # Закрываем рекордеры при выходе из цикла
            if speaker_rec:
                speaker_rec.close()
            if mic_rec:
                mic_rec.close()

        except Exception as e:
            self.signals.status_updated.emit(f"Критическая ошибка записи: {e}")
            print(f"Критическая ошибка записи: {e}")
            import traceback
            traceback.print_exc()

    def get_text_for_period(self, seconds):
        """Получает текст за указанный период времени"""
        return self.text_buffer.get_text(seconds)

    def stop(self):
        self.running = False


# Класс оверлея с улучшенным интерфейсом
class OverlayWindow(QWidget):
    def __init__(self, audio_processor):
        super().__init__()
        self.audio_processor = audio_processor
        self.compact_mode = True  # Режим компактного отображения

        # Настройка окна как оверлея
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Создаем главный вертикальный layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Создаем компактный интерфейс
        self.setup_compact_ui()

        # Для перетаскивания окна
        self.oldPos = None

        # Подключаем сигналы
        self.audio_processor.signals.text_updated.connect(self.update_text)
        self.audio_processor.signals.status_updated.connect(self.update_status)
        self.audio_processor.signals.audio_level_updated.connect(self.update_audio_level)

        # Таймер для автоматического скрытия статуса
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.hide_status)

        # Начальная позиция
        self.setGeometry(100, 100, 500, 100)

    def setup_compact_ui(self):
        # Очищаем текущий layout
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Создаем горизонтальный layout для текста и кнопок
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
        self.label.setWordWrap(True)
        h_layout.addWidget(self.label, 1)

        # Создаем вертикальный layout для кнопок
        buttons_layout = QVBoxLayout()

        # Кнопка копирования
        self.copy_button = QPushButton("Копировать")
        self.copy_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 180, 200);
            border-radius: 10px;
            padding: 5px;
            margin-left: 5px;
        """)
        self.copy_button.setFixedWidth(120)
        self.copy_button.clicked.connect(self.copy_text)
        buttons_layout.addWidget(self.copy_button)

        # Кнопка настроек
        self.settings_button = QPushButton("⚙️")
        self.settings_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 60, 200);
            border-radius: 10px;
            padding: 5px;
            margin-left: 5px;
        """)
        self.settings_button.setFixedWidth(40)
        self.settings_button.clicked.connect(self.toggle_ui_mode)
        buttons_layout.addWidget(self.settings_button)

        h_layout.addLayout(buttons_layout)
        self.main_layout.addLayout(h_layout)

        # Статусная строка (скрыта по умолчанию)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            font-size: 14px; 
            color: #cccccc; 
            background-color: rgba(0, 0, 0, 100);
            border-radius: 5px;
            padding: 5px;
            margin-top: 5px;
        """)
        self.status_label.setVisible(False)
        self.main_layout.addWidget(self.status_label)

        # Индикатор уровня звука
        self.level_indicator = QSlider(Qt.Horizontal)
        self.level_indicator.setRange(0, 100)
        self.level_indicator.setValue(0)
        self.level_indicator.setEnabled(False)
        self.level_indicator.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 200, 0, 200);
                width: 8px;
                margin: -2px 0;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 200, 0, 150);
                border-radius: 4px;
            }
        """)
        self.level_indicator.setFixedHeight(15)
        self.main_layout.addWidget(self.level_indicator)

        self.setLayout(self.main_layout)

    def setup_extended_ui(self):
        # Очищаем текущий layout
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Создаем метку с текстом
        self.label = QLabel("Ожидание текста...")
        self.label.setStyleSheet("""
            font-size: 20px; 
            color: white; 
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 10px;
        """)
        self.label.setWordWrap(True)
        self.main_layout.addWidget(self.label)

        # Индикатор уровня звука
        self.level_indicator = QSlider(Qt.Horizontal)
        self.level_indicator.setRange(0, 100)
        self.level_indicator.setValue(0)
        self.level_indicator.setEnabled(False)
        self.level_indicator.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 200, 0, 200);
                width: 8px;
                margin: -2px 0;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 200, 0, 150);
                border-radius: 4px;
            }
        """)
        self.main_layout.addWidget(self.level_indicator)

        # Настройки источников звука
        sources_layout = QHBoxLayout()

        # Выбор микрофона
        self.mic_check = QPushButton("🎤")
        self.mic_check.setCheckable(True)
        self.mic_check.setChecked(self.audio_processor.use_mic)
        self.mic_check.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: white;
                background-color: rgba(60, 60, 60, 200);
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: rgba(0, 120, 0, 200);
            }
        """)
        self.mic_check.clicked.connect(self.toggle_mic)
        sources_layout.addWidget(self.mic_check)

        # Громкость микрофона
        self.mic_volume = QSlider(Qt.Horizontal)
        self.mic_volume.setRange(0, 100)
        self.mic_volume.setValue(int(self.audio_processor.mic_volume * 100))
        self.mic_volume.valueChanged.connect(self.update_mic_volume)
        self.mic_volume.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 120, 0, 200);
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 120, 0, 150);
                border-radius: 4px;
            }
        """)
        sources_layout.addWidget(self.mic_volume)

        # Выбор системного звука
        self.speaker_check = QPushButton("🔊")
        self.speaker_check.setCheckable(True)
        self.speaker_check.setChecked(self.audio_processor.use_speaker)
        self.speaker_check.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: white;
                background-color: rgba(60, 60, 60, 200);
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: rgba(0, 0, 120, 200);
            }
        """)
        self.speaker_check.clicked.connect(self.toggle_speaker)
        sources_layout.addWidget(self.speaker_check)

        # Громкость системного звука
        self.speaker_volume = QSlider(Qt.Horizontal)
        self.speaker_volume.setRange(0, 100)
        self.speaker_volume.setValue(int(self.audio_processor.speaker_volume * 100))
        self.speaker_volume.valueChanged.connect(self.update_speaker_volume)
        self.speaker_volume.setStyleSheet("""
            QSlider::groove:horizontal {
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgba(0, 0, 120, 200);
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 0, 120, 150);
                border-radius: 4px;
            }
        """)
        sources_layout.addWidget(self.speaker_volume)

        self.main_layout.addLayout(sources_layout)

        # Кнопки действий
        buttons_layout = QHBoxLayout()

        # Кнопка копирования
        self.copy_button = QPushButton("Копировать (30 сек)")
        self.copy_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 180, 200);
            border-radius: 10px;
            padding: 5px;
        """)
        self.copy_button.clicked.connect(self.copy_text)
        buttons_layout.addWidget(self.copy_button)

        # Кнопка копирования длинного текста
        self.copy_long_button = QPushButton("Копировать (2 мин)")
        self.copy_long_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 180, 200);
            border-radius: 10px;
            padding: 5px;
        """)
        self.copy_long_button.clicked.connect(lambda: self.copy_text(120))
        buttons_layout.addWidget(self.copy_long_button)

        # Кнопка свернуть
        self.compact_button = QPushButton("Свернуть")
        self.compact_button.setStyleSheet("""
            font-size: 16px;
            color: white;
            background-color: rgba(60, 60, 60, 200);
            border-radius: 10px;
            padding: 5px;
        """)
        self.compact_button.clicked.connect(self.toggle_ui_mode)
        buttons_layout.addWidget(self.compact_button)

        self.main_layout.addLayout(buttons_layout)

        # Статусная строка
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            font-size: 14px; 
            color: #cccccc; 
            background-color: rgba(0, 0, 0, 100);
            border-radius: 5px;
            padding: 5px;
            margin-top: 5px;
        """)
        self.status_label.setVisible(False)
        self.main_layout.addWidget(self.status_label)

        self.setLayout(self.main_layout)

    def toggle_ui_mode(self):
        self.compact_mode = not self.compact_mode
        if self.compact_mode:
            self.setup_compact_ui()
        else:
            self.setup_extended_ui()

        # Обновляем текст
        display_text = self.audio_processor.get_text_for_period(5)
        self.update_text(display_text)

    def toggle_mic(self):
        self.audio_processor.use_mic = self.mic_check.isChecked()
        self.update_status(f"Микрофон {'включен' if self.audio_processor.use_mic else 'выключен'}")

    def toggle_speaker(self):
        self.audio_processor.use_speaker = self.speaker_check.isChecked()
        self.update_status(f"Системный звук {'включен' if self.audio_processor.use_speaker else 'выключен'}")

    def update_mic_volume(self, value):
        self.audio_processor.mic_volume = value / 100.0

    def update_speaker_volume(self, value):
        self.audio_processor.speaker_volume = value / 100.0

    def copy_text(self, seconds=30):
        full_text = self.audio_processor.get_text_for_period(seconds)
        print(f"\n=== Текст за последние {seconds} секунд ===")
        print(full_text)
        print("===================================\n")

        # Копируем текст в буфер обмена
        clipboard = QApplication.clipboard()
        clipboard.setText(full_text)
        self.update_status("Текст скопирован в буфер обмена")

    def update_text(self, text):
        if text:
            self.label.setText(text)
        else:
            self.label.setText("Ожидание текста...")
        # Изменяем размер окна в зависимости от текста
        self.adjustSize()

    def update_status(self, text):
        self.status_label.setText(text)
        self.status_label.setVisible(True)
        # Запускаем таймер для скрытия статуса через 3 секунды
        self.status_timer.start(3000)

    def hide_status(self):
        self.status_label.setVisible(False)
        self.status_timer.stop()

    def update_audio_level(self, level):
        # Обновляем индикатор уровня звука
        self.level_indicator.setValue(int(level * 100))

        # Меняем цвет индикатора в зависимости от уровня
        if level > 0.8:
            color = "rgba(200, 0, 0, 150)"  # Красный при перегрузке
        elif level > 0.5:
            color = "rgba(200, 200, 0, 150)"  # Желтый при среднем уровне
        else:
            color = "rgba(0, 200, 0, 150)"  # Зеленый при нормальном уровне

        self.level_indicator.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius:
            QSlider::groove:horizontal {{
                background: rgba(50, 50, 50, 150);
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {color};
                width: 8px;
                margin: -2px 0;
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border-radius: 4px;
            }}
        """)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_L and event.modifiers() & Qt.AltModifier:
            self.copy_text()
        elif event.key() == Qt.Key_Escape:
            # Переключаемся в компактный режим при нажатии Escape
            if not self.compact_mode:
                self.toggle_ui_mode()

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

    # Устанавливаем путь к плагинам, если необходимо



    # Создаем и запускаем аудио процессор
    audio_processor = AudioProcessor(MODEL_PATH)
    audio_processor.start()

    # Создаем оверлей
    overlay = OverlayWindow(audio_processor)
    overlay.show()

    sys.exit(app.exec_())