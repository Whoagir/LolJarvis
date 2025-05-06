import threading
import queue
import time
import numpy as np
import soundcard as sc
import whisper
from PyQt5.QtCore import pyqtSignal, QObject
import gc  # Добавляем сборщик мусора

# Константы
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио
MAX_AUDIO_SECONDS = 15  # Уменьшаем с 30 до 10 секунд
MAX_SAMPLES = int(MAX_AUDIO_SECONDS * RATE)


class Signals(QObject):
    text_updated = pyqtSignal(str)


class RecognizedSegment:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return self.text


class AudioProcessor(threading.Thread):
    def __init__(self, model_name="small"):  # Используем tiny модель для экономии памяти
        super().__init__()
        self.daemon = True
        self.audio_queue = queue.Queue()
        self.signals = Signals()
        self.running = True
        self.model = whisper.load_model(model_name)
        self.segments_dict = {}
        self.last_update_time = time.time()
        self.last_recognition_time = time.time()
        self.audio_buffer = []
        self.recognition_start_time = None

    def run(self):
        audio_thread = threading.Thread(target=self.record_audio)
        audio_thread.daemon = True
        audio_thread.start()

        while self.running:
            try:
                # Обрабатываем очередь, но не накапливаем слишком много данных
                while not self.audio_queue.empty() and len(self.audio_buffer) < MAX_AUDIO_SECONDS * 4:
                    try:
                        audio_data, timestamp = self.audio_queue.get(timeout=0.1)
                        self.audio_buffer.append((audio_data, timestamp))
                    except queue.Empty:
                        break

                current_time = time.time()

                # Обрабатываем каждые 5 секунд
                if current_time - self.last_recognition_time > 5 and self.audio_buffer:
                    # Берем только последние 5 секунд аудио
                    cutoff_time = current_time - 5
                    recent_audio = []
                    recent_timestamps = []

                    for chunk, ts in self.audio_buffer:
                        if ts > cutoff_time:
                            recent_audio.append(chunk)
                            recent_timestamps.append(ts)

                    if recent_audio:
                        try:
                            # Создаем временный файл для аудио вместо обработки в памяти
                            import tempfile
                            import soundfile as sf

                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                                temp_filename = temp_file.name

                            audio_array = np.concatenate(recent_audio)
                            # Нормализуем аудио
                            if np.max(np.abs(audio_array)) > 0:
                                audio_array = audio_array / np.max(np.abs(audio_array))

                            # Сохраняем во временный файл
                            sf.write(temp_filename, audio_array, RATE)

                            # Освобождаем память
                            del audio_array
                            gc.collect()

                            # Транскрибируем из файла
                            result = self.model.transcribe(
                                temp_filename,
                                language="ru",
                                beam_size=1,
                                fp16=False,
                                temperature=0  # Уменьшает вариативность и потребление памяти
                            )

                            # Удаляем временный файл
                            import os
                            os.unlink(temp_filename)

                            t_start = min(recent_timestamps) if recent_timestamps else current_time - 5

                            for segment in result["segments"]:
                                start = t_start + segment["start"]
                                end = t_start + segment["end"]
                                text = segment["text"]
                                while start in self.segments_dict:
                                    start += 0.000001
                                self.segments_dict[start] = RecognizedSegment(text, start, end)

                            # Очистка старых сегментов
                            cutoff_time = current_time - 30  # Храним историю только за 30 секунд
                            self.segments_dict = {k: v for k, v in self.segments_dict.items() if
                                                  v.end_time > cutoff_time}

                            # Обновление текста
                            display_text = self.get_display_text()
                            self.signals.text_updated.emit(display_text)

                            # Принудительно освобождаем память
                            gc.collect()

                        except Exception as e:
                            print(f"Ошибка при обработке аудио: {e}")
                            import traceback
                            traceback.print_exc()

                        self.last_recognition_time = current_time

                    # Очистка старого аудио из буфера - важно для предотвращения утечек памяти
                    cutoff_buffer_time = current_time - MAX_AUDIO_SECONDS
                    self.audio_buffer = [(chunk, ts) for chunk, ts in self.audio_buffer if ts > cutoff_buffer_time]

                # Небольшая пауза для снижения нагрузки на CPU
                time.sleep(0.1)

            except Exception as e:
                print(f"Ошибка распознавания: {e}")
                import traceback
                traceback.print_exc()

    def get_display_text(self):
        current_time = time.time()
        cutoff_time = current_time - 5
        recent_segments = sorted([seg for seg in self.segments_dict.values() if seg.start_time > cutoff_time],
                                 key=lambda x: x.start_time)
        display_text = " ".join([seg.text for seg in recent_segments])
        return display_text.strip()

    def get_text_for_period(self, seconds):
        current_time = time.time()
        cutoff_time = current_time - seconds
        period_segments = sorted([seg for seg in self.segments_dict.values() if seg.start_time >= cutoff_time],
                                 key=lambda x: x.start_time)
        full_text = " ".join([seg.text for seg in period_segments])
        return full_text.strip()

    def record_audio(self):
        try:
            self.recognition_start_time = time.time()
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
            default_mic = sc.default_microphone()

            print("Запись звука началась...")

            with loopback_mic.recorder(samplerate=RATE, channels=CHANNELS) as speaker_rec, \
                    default_mic.recorder(samplerate=RATE, channels=CHANNELS) as mic_rec:
                while self.running:
                    speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                    mic_data = mic_rec.record(numframes=CHUNK_SIZE)
                    mixed_data = np.mean([speaker_data, mic_data], axis=0).astype(np.float32)

                    # Добавляем только если очередь не слишком большая
                    if self.audio_queue.qsize() < 100:  # Ограничиваем размер очереди
                        self.audio_queue.put((mixed_data, time.time()))

                    # Небольшая пауза для снижения нагрузки
                    time.sleep(0.01)

        except Exception as e:
            print(f"Ошибка записи звука: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False