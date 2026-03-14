import queue
import threading
import time

import numpy as np

from src.decoder.features_extract.feature_extract import extract_feature
from src.decoder.online_inference.ml_decoder.ml_decoder import decode
from src.decoder.preprocess.preprocessor import preprocess_data

class DecoderThread(threading.Thread):
    # 解码线程：接收 ndarray，完成预处理+特征提取+解码
    def __init__(self, fs, decoder_cfg, feature_cfg, model_bundle, name="DecoderThread"):
        super().__init__(name=name, daemon=True)
        self.fs = fs
        self.decoder_cfg = decoder_cfg
        self.feature_cfg = feature_cfg
        self.model_bundle = model_bundle

        self._stop_event = threading.Event()
        self._input_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._busy_lock = threading.Lock()
        self._is_busy = False

    def _set_busy(self, value):
        with self._busy_lock:
            self._is_busy = value

    def is_busy(self):
        with self._busy_lock:
            return self._is_busy

    def submit(self, decode_id, data, timestamp_s):
        payload = {
            "decode_id": int(decode_id),
            "timestamp_s": float(timestamp_s),
            "data": np.asarray(data, dtype=float),
        }

        while True:
            try:
                self._input_queue.put_nowait(payload)
                return True
            except queue.Full:
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    return False

    def consume_result(self):
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def decode_stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._set_busy(True)

            try:
                decode_id = payload["decode_id"]
                data_timestamp_s = payload["timestamp_s"]
                data = payload["data"]

                preprocess_start = time.perf_counter()
                preprocessed = preprocess_data(data, self.fs, self.decoder_cfg)
                preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

                decode_start = time.perf_counter()
                featured = extract_feature(preprocessed, self.fs, self.feature_cfg)
                decode_result = decode(featured, self.model_bundle)
                decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

                result_payload = {
                    "decode_id": decode_id,
                    "data_received_time_s": round(data_timestamp_s, 6),
                    "data_shape": tuple(data.shape),
                    "preprocess_time_ms": round(preprocess_time_ms, 3),
                    "decode_time_ms": round(decode_time_ms, 3),
                    "result": decode_result,
                }

            except Exception as e:
                result_payload = {
                    "decode_id": payload.get("decode_id", -1),
                    "data_received_time_s": round(payload.get("timestamp_s", 0.0), 6),
                    "data_shape": tuple(payload.get("data", np.array([])).shape),
                    "preprocess_time_ms": None,
                    "decode_time_ms": None,
                    "result": {
                        "predicted_target": None,
                        "confidence": None,
                        "probabilities": {},
                        "success": False,
                        "method": "ml_decoder",
                        "error": str(e),
                    },
                }

            while True:
                try:
                    self._result_queue.put_nowait(result_payload)
                    break
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        break

            self._set_busy(False)


class StimThread(threading.Thread):
    # 刺激线程：接收一个 bool，决定是否发刺激，并控制持续时间后停止
    def __init__(self, stimulator=None, name="StimThread"):
        super().__init__(name=name, daemon=True)
        self.stimulator = stimulator

        self._stop_event = threading.Event()
        self._command_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=10)
        self._busy_lock = threading.Lock()
        self._is_busy = False

    def _set_busy(self, value):
        with self._busy_lock:
            self._is_busy = value

    def is_busy(self):
        with self._busy_lock:
            return self._is_busy

    def submit(self, decode_id, should_stim, params=None, duration_ms=500, command_label=""):
        payload = {
            "decode_id": int(decode_id),
            "should_stim": bool(should_stim),
            "params": params,
            "duration_ms": int(duration_ms),
            "command_label": command_label,
        }

        while True:
            try:
                self._command_queue.put_nowait(payload)
                return True
            except queue.Full:
                try:
                    self._command_queue.get_nowait()
                except queue.Empty:
                    return False

    def consume_result(self):
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def stim_stop(self):
        self._stop_event.set()
        try:
            if self.stimulator is not None:
                self.stimulator.stop_stimulation()
        except Exception:
            pass

    def run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._command_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._set_busy(True)

            decode_id = payload["decode_id"]
            should_stim = payload["should_stim"]
            params = payload["params"]
            duration_ms = payload["duration_ms"]
            command_label = payload["command_label"]

            command_sent = 0
            error_message = ""

            try:
                # 只有 should_stim=True 且刺激器可用时才真的发刺激
                if should_stim and self.stimulator is not None and params is not None:
                    command_sent = int(
                        bool(
                            self.stimulator.send_command_with_duration(
                                params=params,
                                duration_ms=duration_ms,
                            )
                        )
                    )

                    # 持续刺激指定时间后，主动停止
                    if command_sent:
                        deadline = time.perf_counter() + duration_ms / 1000.0
                        while (not self._stop_event.is_set()) and (time.perf_counter() < deadline):
                            time.sleep(0.005)

                        try:
                            self.stimulator.stop_stimulation()
                        except Exception:
                            pass

            except Exception as e:
                error_message = str(e)
                command_sent = 0

            result_payload = {
                "decode_id": decode_id,
                "command_sent": command_sent,
                "command_content": command_label if command_sent else "",
                "error": error_message,
            }

            while True:
                try:
                    self._result_queue.put_nowait(result_payload)
                    break
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        break

            self._set_busy(False)