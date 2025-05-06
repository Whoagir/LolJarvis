import soundcard as sc
import numpy as np
import wave

# Настройки
RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "../../../test_audio.wav"

# Получаем устройство вывода
speaker = sc.default_speaker()

# Получаем loopback устройство для захвата звука с колонок
try:
    mic = sc.get_microphone(speaker.id, include_loopback=True)
except Exception as e:
    print(f"Ошибка при получении loopback: {e}")
    exit(1)

# Начинаем запись
print("Запись началась...")
with mic.recorder(samplerate=RATE, channels=CHANNELS) as rec:
    data = rec.record(numframes=RATE * RECORD_SECONDS)
    print("Запись завершена.")

# Преобразуем данные в 16-битный формат
data = (data * 32767).astype(np.int16)

# Сохраняем в WAV-файл
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(data.tobytes())

print(f"Аудио сохранено в {WAVE_OUTPUT_FILENAME}")