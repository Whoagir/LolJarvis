import threading
import queue
import time
import numpy as np
import soundcard as sc
import whisper
import tempfile
import soundfile as sf
import os
from PyQt5.QtCore import pyqtSignal, QObject, QThread

# Константы
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = RATE // 4  # 0.25 секунды аудио


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

            # Нормализуем аудио
            if np.max(np.abs(self.audio_data)) > 0:
                self.audio_data = self.audio_data / np.max(np.abs(self.audio_data))

            # Сохраняем во временный файл
            sf.write(temp_filename, self.audio_data, RATE)

            # Эмулируем прогресс (т.к. Whisper не дает прогресс напрямую)
            for i in range(10):
                self.progress.emit(i * 10)
                if i < 9:  # Не спим на последней итерации
                    time.sleep(0.2)

            # Транскрибируем из файла
            result = self.model.transcribe(
                temp_filename,
                language="ru",
                beam_size=1,
                fp16=False,
                temperature=0
            )

            # Удаляем временный файл
            os.unlink(temp_filename)

            # Собираем текст из всех сегментов
            full_text = " ".join([segment["text"] for segment in result["segments"]])

            self.progress.emit(100)
            self.result.emit(full_text)

        except Exception as e:
            print(f"Ошибка при транскрибации: {e}")
            import traceback
            traceback.print_exc()
            self.result.emit(f"Ошибка транскрибации: {str(e)}")


class AudioRecorderSignals(QObject):
    transcription_complete = pyqtSignal(str)
    transcription_progress = pyqtSignal(int)


class AudioRecorder:
    def __init__(self, model_name="small"):
        self.signals = AudioRecorderSignals()
        self.running = False
        self.recording = False
        self.audio_buffer = []
        self.model = whisper.load_model(model_name)
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

            with loopback_mic.recorder(samplerate=RATE, channels=CHANNELS) as speaker_rec, \
                    default_mic.recorder(samplerate=RATE, channels=CHANNELS) as mic_rec:
                while self.running:
                    if self.recording:
                        speaker_data = speaker_rec.record(numframes=CHUNK_SIZE)
                        mic_data = mic_rec.record(numframes=CHUNK_SIZE)
                        mixed_data = np.mean([speaker_data, mic_data], axis=0).astype(np.float32)
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

        # Создаем и запускаем рабочий поток для транскрибации
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