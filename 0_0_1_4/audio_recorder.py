import threading
import time
import numpy as np
import soundcard as sc
import whisper
import tempfile
import soundfile as sf
import os
import re
import torch
from scipy import signal
from PyQt5.QtCore import pyqtSignal, QObject, QThread

# Константы
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио
MAX_SEGMENT_LENGTH = 30 * RATE  # 30 секунд для разделения длинных аудио


def preprocess_audio(audio_data, sample_rate=RATE):
    """Улучшенная предобработка аудио для лучшего распознавания речи"""
    # Нормализация
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    # Удаление постоянной составляющей
    audio_data = audio_data - np.mean(audio_data)

    # Применение предусиления высоких частот для улучшения разборчивости речи
    b, a = signal.butter(2, 300 / (sample_rate / 2), 'highpass')
    audio_data = signal.lfilter(b, a, audio_data)

    # Компрессия динамического диапазона для выравнивания громкости
    threshold = 0.1
    ratio = 0.5
    audio_data_compressed = np.copy(audio_data)
    mask = np.abs(audio_data) > threshold
    audio_data_compressed[mask] = threshold + (np.abs(audio_data[mask]) - threshold) * ratio
    audio_data_compressed = audio_data_compressed * np.sign(audio_data)

    # Нормализация после обработки
    if np.max(np.abs(audio_data_compressed)) > 0:
        audio_data_compressed = audio_data_compressed / np.max(np.abs(audio_data_compressed))

    return audio_data_compressed


def postprocess_transcription(text):
    """Улучшение качества транскрибированного текста"""
    # Удаление повторяющихся слов
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # Исправление пунктуации
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)

    # Капитализация первой буквы предложения
    text = re.sub(r'([.!?]\s+)([a-zа-я])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = text[0].upper() + text[1:] if text else text

    return text


class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str)

    def __init__(self, audio_data, model):
        super().__init__()
        self.audio_data = audio_data
        self.model = model

    def run(self):
        try:
            # Создаем временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name

            # Применяем предобработку аудио
            processed_audio = preprocess_audio(self.audio_data)

            # Сохраняем во временный файл
            sf.write(temp_filename, processed_audio, RATE)

            # Эмулируем прогресс (т.к. Whisper не дает прогресс напрямую)
            for i in range(10):
                self.progress.emit(i * 10)
                if i < 9:  # Не спим на последней итерации
                    time.sleep(0.1)  # Уменьшаем задержку для более быстрого отклика

            # Улучшенные параметры транскрибации
            result = self.model.transcribe(
                temp_filename,
                language="ru",
                beam_size=5,  # Увеличиваем для лучшего поиска
                fp16=torch.cuda.is_available(),  # Используем fp16 если доступен GPU
                temperature=0.2,  # Небольшая температура для более стабильных результатов
                initial_prompt="Это транскрипция разговора на русском языке."  # Добавляем контекст
            )

            # Удаляем временный файл
            os.unlink(temp_filename)

            # Собираем текст из всех сегментов
            full_text = " ".join([segment["text"] for segment in result["segments"]])

            # Применяем постобработку
            full_text = postprocess_transcription(full_text)

            self.progress.emit(100)
            self.result.emit(full_text)

        except Exception as e:
            print(f"Ошибка при транскрибации: {e}")
            import traceback
            traceback.print_exc()
            self.result.emit(f"Ошибка транскрибации: {str(e)}")


class SegmentTranscriptionWorker(QThread):
    """Рабочий поток для транскрибации длинных аудио по сегментам"""
    progress = pyqtSignal(int)
    result = pyqtSignal(str)

    def __init__(self, audio_segments, model):
        super().__init__()
        self.audio_segments = audio_segments
        self.model = model

    def run(self):
        try:
            all_results = []

            for i, segment in enumerate(self.audio_segments):
                # Обновляем прогресс
                self.progress.emit(int((i / len(self.audio_segments)) * 100))

                # Создаем временный файл для сегмента
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name

                # Предобработка и сохранение
                processed_segment = preprocess_audio(segment)
                sf.write(temp_filename, processed_segment, RATE)

                # Транскрибируем
                result = self.model.transcribe(
                    temp_filename,
                    language="ru",
                    beam_size=5,
                    fp16=torch.cuda.is_available(),
                    temperature=0.2,
                    initial_prompt="Это транскрипция разговора на русском языке."
                )

                # Удаляем временный файл
                os.unlink(temp_filename)

                # Добавляем результат
                segment_text = " ".join([s["text"] for s in result["segments"]])
                all_results.append(segment_text)

            # Объединяем результаты
            full_text = " ".join(all_results)
            full_text = postprocess_transcription(full_text)

            self.progress.emit(100)
            self.result.emit(full_text)

        except Exception as e:
            print(f"Ошибка при транскрибации сегментов: {e}")
            import traceback
            traceback.print_exc()
            self.result.emit(f"Ошибка транскрибации: {str(e)}")


class AudioRecorderSignals(QObject):
    transcription_complete = pyqtSignal(str)
    transcription_progress = pyqtSignal(int)


class AudioRecorder:
    def __init__(self, model_name="medium"):  # Улучшаем модель до medium
        self.signals = AudioRecorderSignals()
        self.running = False
        self.recording = False
        self.audio_buffer = []

        # Используем GPU, если доступен
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Загрузка модели Whisper {model_name} на устройство {device}...")
        self.model = whisper.load_model(model_name, device=device)

        self.record_thread = None
        self.transcription_worker = None

    def start_recording(self):
        if not self.running:
            self.running = True
            self.recording = True
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
        else:
            self.recording = True

    def pause_recording(self):
        self.recording = False

    def resume_recording(self):
        self.recording = True

    def clear_recording(self):
        self.audio_buffer = []

    def has_recording(self):
        return len(self.audio_buffer) > 0

    def _record_audio(self):
        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
            default_mic = sc.default_microphone()

            print("Запись звука началась...")

            # Увеличиваем размер буфера для более стабильной записи
            buffer_size = CHUNK_SIZE * 2

            with loopback_mic.recorder(samplerate=RATE, channels=CHANNELS, blocksize=buffer_size) as speaker_rec, \
                    default_mic.recorder(samplerate=RATE, channels=CHANNELS, blocksize=buffer_size) as mic_rec:
                while self.running:
                    if self.recording:
                        speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                        mic_data = mic_rec.record(numframes=CHUNK_SIZE)

                        # Умное смешивание: используем только тот источник, где есть речь
                        speaker_level = np.max(np.abs(speaker_data))
                        mic_level = np.max(np.abs(mic_data))

                        if speaker_level > 0.05 and speaker_level > mic_level * 1.5:
                            # Используем звук с динамиков
                            mixed_data = speaker_data
                        elif mic_level > 0.05:
                            # Используем звук с микрофона
                            mixed_data = mic_data
                        else:
                            # Смешиваем, если нет явного источника
                            mixed_data = np.mean([speaker_data, mic_data], axis=0)

                        # Преобразуем в float32 для обработки
                        mixed_data = mixed_data.astype(np.float32)

                        # Определяем, содержит ли фрагмент речь (VAD - Voice Activity Detection)
                        if np.max(np.abs(mixed_data)) > 0.02:  # Простой VAD на основе амплитуды
                            self.audio_buffer.append(mixed_data)
                    time.sleep(0.01)

        except Exception as e:
            print(f"Ошибка записи звука: {e}")
            import traceback
            traceback.print_exc()

    def transcribe(self):
        if not self.audio_buffer:
            self.signals.transcription_complete.emit("Нет аудио для транскрибации")
            return

        # Объединяем все фрагменты аудио
        audio_array = np.concatenate(self.audio_buffer)

        # Если аудио длинное, разделяем на сегменты для лучшей транскрибации
        if len(audio_array) > MAX_SEGMENT_LENGTH:
            segments = []
            for i in range(0, len(audio_array), MAX_SEGMENT_LENGTH):
                segment = audio_array[i:i + MAX_SEGMENT_LENGTH]
                segments.append(segment)

            # Транскрибируем каждый сегмент отдельно
            self.transcription_worker = SegmentTranscriptionWorker(segments, self.model)
            self.transcription_worker.progress.connect(self.signals.transcription_progress)
            self.transcription_worker.result.connect(self.signals.transcription_complete)
            self.transcription_worker.start()
        else:
            # Для коротких аудио используем стандартный подход
            self.transcription_worker = TranscriptionWorker(audio_array, self.model)
            self.transcription_worker.progress.connect(self.signals.transcription_progress)
            self.transcription_worker.result.connect(self.signals.transcription_complete)
            self.transcription_worker.start()

    def stop(self):
        self.running = False
        self.recording = False
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=1.0)

        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.terminate()
            self.transcription_worker.wait()