import vosk
import wave
import json
import sys

# Путь к модели Vosk
MODEL_PATH = "C:/model/vosk-model-small-ru-0.22"  # Укажи путь к своей модели
AUDIO_FILE = "../../test_audio.wav"  # Файл из предыдущего шага

# Загружаем модель
model = vosk.Model(MODEL_PATH)

# Открываем аудиофайл
wf = wave.open(AUDIO_FILE, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
    print("Аудиофайл должен быть моно, 16 бит, 16000 Гц")
    sys.exit(1)

# Инициализируем распознаватель
rec = vosk.KaldiRecognizer(model, wf.getframerate())

# Читаем аудио и распознаём
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print("Текст:", result.get("text", ""))

# Финальный результат
final_result = json.loads(rec.FinalResult())
print("Финальный текст:", final_result.get("text", ""))