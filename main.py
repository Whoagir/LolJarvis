import os
import sys
import warnings
from PyQt5.QtWidgets import QApplication
from audio_processor_vosk import AudioProcessor
from gui import OverlayWindow
from soundcard.mediafoundation import SoundcardRuntimeWarning  # Добавлен импорт

# Отключаем предупреждения SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

# Указываем путь к папке platforms
plugins_path = r'C:\Users\Пользователь\PycharmProjects\help tech sob\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path

# Путь к модели Vosk
MODEL_PATH = "C:/model/vosk-model-small-ru-0.22"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    audio_processor = AudioProcessor(MODEL_PATH)
    audio_processor.start()
    overlay = OverlayWindow(audio_processor)
    overlay.show()
    sys.exit(app.exec_())