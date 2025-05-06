import threading, queue, time, json, collections
from datetime import datetime

import numpy as np
import soundcard as sc
import vosk
from PyQt5.QtCore import pyqtSignal, QObject

RATE        = 16_000
CHANNELS    = 1
CHUNK_SIZE  = RATE // 4          # 0.25 c
PARTIAL_GUI_DT = 0.25           # не чаще, c
WINDOW_SEC  = 30

# -----------------------------------------------------
class Signals(QObject):
    text_updated = pyqtSignal(str)   # для GUI
    debug_log    = pyqtSignal(str)   # в консоль

# -----------------------------------------------------
class Word:
    __slots__ = ('txt', 'ts')
    def __init__(self, txt, ts): self.txt, self.ts = txt, ts

# -----------------------------------------------------
class AudioProcessor(threading.Thread):
    def __init__(self, model_path: str):
        super().__init__(daemon=True)
        self.signals   = Signals()
        self.model     = vosk.Model(model_path)
        self.rec       = vosk.KaldiRecognizer(self.model, RATE)
        self.rec.SetWords(True)

        self.audio_q   = queue.Queue()
        self.running   = True

        self.words     = collections.deque()   # deque[Word]
        self.partial   = ''
        self.last_gui  = 0.0
        self.seen_total_words = 0             # diff-счетчик

    # --------------  public API  ----------------------
    def get_text_for_period(self, sec: int) -> str:
        now = time.time()
        return ' '.join(w.txt for w in self.words if w.ts >= now-sec)

    def stop(self):
        self.running = False

    # --------------  thread run  ----------------------
    def run(self):
        threading.Thread(target=self._record, daemon=True).start()
        while self.running:
            try:
                data = self.audio_q.get(timeout=1)
            except queue.Empty:
                continue

            now = time.time()
            if self.rec.AcceptWaveform(data):
                res = json.loads(self.rec.Result())
                self._append_final(res, now)
                self._update_gui(now, force=True)
            else:
                part = json.loads(self.rec.PartialResult())['partial']
                self.partial = part
                self._update_gui(now)
    # --------------------------------------------------
    def _append_final(self, res: dict, wall_time: float):
        """
        Добавляем ТОЛЬКО новые слова из final-результата.
        start/end внутри res идут в секундах С МОМЕНТА СТАРТА recognizer’а,
        поэтому приводим их к абсолютному времени «сейчас».
        """
        words_json = res.get('result', [])
        if not words_json:
            return

        # отсекаем уже увиденные
        new_words = words_json[self.seen_total_words:]
        self.seen_total_words = len(words_json)

        for w in new_words:
            ts = wall_time - (words_json[-1]['end'] - w['start'])
            self.words.append(Word(w['word'], ts))

        # чистим окно 30 с
        cutoff = wall_time - WINDOW_SEC
        while self.words and self.words[0].ts < cutoff:
            self.words.popleft()

        self.partial = ''   # partial обнуляем после финала

    # --------------------------------------------------
    def _update_gui(self, now: float, force=False):
        if not force and now - self.last_gui < PARTIAL_GUI_DT:
            return
        self.last_gui = now

        text = ' '.join(w.txt for w in self.words)
        if self.partial:
            text = f'{text} {self.partial}'.strip()

        self.signals.text_updated.emit(text)
        stamp = datetime.fromtimestamp(now).strftime('%H:%M:%S')
        self.signals.debug_log.emit(f'[{stamp}] {text}')

    # --------------  audio capture  -------------------
    def _record(self):
        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(speaker.id, include_loopback=True)
            mic      = sc.default_microphone()

            with loopback.recorder(RATE, CHANNELS) as sp_rec,\
                 mic     .recorder(RATE, CHANNELS) as mic_rec:
                self.signals.debug_log.emit('Запись звука началась…')
                while self.running:
                    sp = sp_rec.record(CHUNK_SIZE)
                    mc = mic_rec.record(CHUNK_SIZE)
                    data = self._mix(sp, mc)
                    self.audio_q.put(data)
        except Exception as e:
            self.signals.debug_log.emit(f'Ошибка записи: {e}')

    # --------------------------------------------------
    @staticmethod
    def _mix(a: np.ndarray, b: np.ndarray) -> bytes:
        """
        Быстро складываем два float32-массива, нормируем, отдаем int16 bytes.
        """
        mix = a + b
        maxv = np.max(np.abs(mix)) or 1.0
        mix = (mix / maxv * 32767).astype(np.int16)
        return mix.tobytes()