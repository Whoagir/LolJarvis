import os
import sys
import warnings
from PyQt5.QtWidgets import QApplication
from audio_recorder import AudioRecorder
from gui import TranscriptionWindow
from soundcard.mediafoundation import SoundcardRuntimeWarning

# Отключаем предупреждения SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

# Указываем путь к папке platforms (измените на свой путь)
plugins_path = r'C:\Users\Пользователь\PycharmProjects\help tech sob\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    audio_recorder = AudioRecorder()
    window = TranscriptionWindow(audio_recorder)
    window.show()
    sys.exit(app.exec_())