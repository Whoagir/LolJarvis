import threading
import queue
import time
import numpy as np
import soundcard as sc
import vosk
from PyQt5.QtCore import pyqtSignal, QObject
import json
from datetime import datetime

# Константы
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио
DISPLAY_WINDOW = 30  # Показывать последние 30 секунд транскрибации


class Signals(QObject):
    text_updated = pyqtSignal(str)
    debug_log = pyqtSignal(str)


class RecognizedSegment:
    def __init__(self, text, timestamp):
        self.text = text
        self.timestamp = timestamp  # Используем один timestamp для упрощения

    def __str__(self):
        return self.text


class AudioProcessor(threading.Thread):
    def __init__(self, model_path):
        super().__init__()
        self.daemon = True
        self.audio_queue = queue.Queue()
        self.signals = Signals()
        self.running = True
        self.model_path = model_path
        self.segments = []  # Список сегментов с временными метками
        self.last_update_time = time.time()
        self.partial_text = ""
        self.current_text = ""  # Текущий полный текст для отображения

    def run(self):
        model = vosk.Model(self.model_path)
        rec = vosk.KaldiRecognizer(model, RATE)
        rec.SetWords(True)

        audio_thread = threading.Thread(target=self.record_audio)
        audio_thread.daemon = True
        audio_thread.start()

        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                current_time = time.time()

                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    if "text" in result and result["text"].strip():
                        # Добавляем новый сегмент с текущим временем
                        segment = RecognizedSegment(result["text"], current_time)
                        self.segments.append(segment)

                        # Обновляем отображаемый текст
                        self.update_display_text()

                        # Логируем для отладки
                        timestamp_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
                        self.signals.debug_log.emit(f"[{timestamp_str}] {self.current_text}")
                else:
                    partial_result = json.loads(rec.PartialResult())
                    self.partial_text = partial_result.get("partial", "")

                    # Обновляем текст с частичным результатом не чаще чем раз в 0.3 секунды
                    if current_time - self.last_update_time > 0.3 and self.partial_text:
                        self.update_display_text()
                        timestamp_str = datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
                        self.signals.debug_log.emit(f"[{timestamp_str}] {self.current_text}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка распознавания: {e}")
                import traceback
                traceback.print_exc()

    def update_display_text(self):
        current_time = time.time()
        self.last_update_time = current_time

        # Очищаем старые сегменты (старше DISPLAY_WINDOW секунд)
        cutoff_time = current_time - DISPLAY_WINDOW
        self.segments = [seg for seg in self.segments if seg.timestamp > cutoff_time]

        # Формируем текст из последних сегментов
        recent_texts = [seg.text for seg in self.segments]
        display_text = " ".join(recent_texts)

        # Добавляем частичный результат, если есть
        if self.partial_text:
            if display_text:
                display_text += " " + self.partial_text
            else:
                display_text = self.partial_text

        self.current_text = display_text.strip()
        self.signals.text_updated.emit(self.current_text)

    def get_text_for_period(self, seconds):
        current_time = time.time()
        cutoff_time = current_time - seconds
        period_segments = [seg for seg in self.segments if seg.timestamp >= cutoff_time]
        period_texts = [seg.text for seg in period_segments]
        full_text = " ".join(period_texts)
        return full_text.strip()

    def record_audio(self):
        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
            default_mic = sc.default_microphone()

            print("Запись звука началась...")
            self.signals.debug_log.emit("Запись звука началась...")

            with loopback_mic.recorder(samplerate=RATE, channels=CHANNELS) as speaker_rec, \
                    default_mic.recorder(samplerate=RATE, channels=CHANNELS) as mic_rec:
                while self.running:
                    speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                    mic_data = mic_rec.record(numframes=CHUNK_SIZE)
                    mixed_data = np.mean([speaker_data, mic_data], axis=0)
                    audio_data = (mixed_data * 32767).astype(np.int16).tobytes()
                    self.audio_queue.put(audio_data)
                    time.sleep(0.01)  # Небольшая пауза для снижения нагрузки

        except Exception as e:
            print(f"Ошибка записи звука: {e}")
            self.signals.debug_log.emit(f"Ошибка записи звука: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False