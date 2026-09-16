"""Paced packet replay with interruptible waits and bounded display updates.

Architecture follows the owner's radar live.py worker/queue separation. CSI
timestamps replace its constant radar frame rate. Processing never drops packets.
"""
from dataclasses import dataclass
import math
import queue
import threading
import time

from .data import Cancelled, read_observations
from .processing import Processor


@dataclass(frozen=True)
class ReplayMessage:
    kind: str
    time: float = 0.0
    status: str = ''
    points: tuple = ()
    estimate: object = None
    detail: str = ''
    processed: int = 0
    display_drops: int = 0


class ReplayWorker(threading.Thread):
    def __init__(self, dataset, feature_config, speed=1.0, processing_config=None):
        super().__init__(daemon=True, name='csi-replay')
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('Replay speed must be positive and finite')
        self.dataset, self.feature_config = dataset, feature_config
        self.speed, self.processing_config = speed, processing_config
        self.messages = queue.Queue(maxsize=4)
        self.cancel = threading.Event()
        self._condition = threading.Condition()
        self._paused = False
        self._pause_started = None
        self._pause_total = 0.0
        self._clock_start = None
        self.display_drops = 0
        self.processed = 0

    def stop(self):
        self.cancel.set()
        with self._condition:
            self._condition.notify_all()

    def pause(self):
        with self._condition:
            if not self._paused:
                self._paused = True
                self._pause_started = time.monotonic()
            self._condition.notify_all()

    def resume(self):
        with self._condition:
            if self._paused:
                self._pause_total += time.monotonic() - self._pause_started
                self._paused = False
                self._pause_started = None
            self._condition.notify_all()

    def _emit(self, message):
        if self.messages.full():
            try:
                self.messages.get_nowait()
                self.display_drops += 1
            except queue.Empty:
                pass
        self.messages.put_nowait(message)

    def _wait(self, target):
        # Absolute deadlines avoid cumulative drift from decoding/processing.
        with self._condition:
            while not self.cancel.is_set():
                while self._paused and not self.cancel.is_set():
                    self._condition.wait(0.05)
                remaining = target - (time.monotonic() - self._clock_start - self._pause_total)
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.05))
        if self.cancel.is_set():
            raise Cancelled('Replay stopped')

    def run(self):
        try:
            self._emit(ReplayMessage('loading', status='Verifying source hashes and decoding'))
            observations, audit = read_observations(self.dataset, self.feature_config, self.cancel)
            processor = Processor(self.feature_config.mode, self.processing_config)
            previous = observations[0].time
            first = previous
            with self._condition:
                self._clock_start = time.monotonic()
                self._pause_total = 0.0
                if self._paused:
                    self._pause_started = self._clock_start
            last_emit = -math.inf
            for observation in observations:
                self._wait((observation.time - first) / self.speed)
                processor.feed(observation)
                self.processed += 1
                previous = observation.time
                now = time.monotonic()
                if now - last_emit >= 0.05:
                    self._emit(ReplayMessage('frame', observation.time, processor.status,
                                             tuple(processor.history), processor.latest,
                                             processed=self.processed, display_drops=self.display_drops))
                    last_emit = now
            self.dataset.verify(self.cancel)
            self._emit(ReplayMessage('eof', previous, 'Replay finished — no live measurement',
                                     tuple(processor.history), None,
                                     f"{audit['packets']} packets processed; source hashes unchanged.",
                                     self.processed, self.display_drops))
        except Cancelled:
            self._emit(ReplayMessage('stopped', status='Stopped — no live measurement', processed=self.processed))
        except Exception as error:
            self._emit(ReplayMessage('error', status='Replay failed', detail=str(error), processed=self.processed))
