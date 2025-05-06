import pyaudio
import wave

# Настройки
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10  # Записываем 5 секунд для теста
WAVE_OUTPUT_FILENAME = "../../test_audio.wav"

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Открываем поток для записи
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Запись началась...")

frames = []

# Записываем аудио
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Запись завершена.")

# Останавливаем и закрываем поток
stream.stop_stream()
stream.close()
p.terminate()

# Сохраняем в файл
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Аудио сохранено в {WAVE_OUTPUT_FILENAME}")