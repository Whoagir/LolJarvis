import os

# Укажи путь к папке platforms
plugins_path = r'C:\Users\Пользователь\PycharmProjects\help tech sob\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path


from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
window.show()
sys.exit(app.exec_())