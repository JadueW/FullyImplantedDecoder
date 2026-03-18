import queue
import threading
import time

import numpy as np

from src.decoder.features_extract.feature_extract import extract_feature, prepare_feature_plan
from src.decoder.online_inference.ml_decoder.ml_decoder import decode
from src.decoder.preprocess.preprocessor import preprocess_data


def _queue_put_latest(target_queue, payload):
    while True:
        try:
            target_queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return False


class _BaseWorkerThread(threading.Thread):
    def __init__(self, *, name, input_maxsize, result_maxsize):
        super().__init__(name=name, daemon=True)
        self._stop_event = threading.Event()
        self._input_queue = queue.Queue(maxsize=input_maxsize)
        self._result_queue = queue.Queue(maxsize=result_maxsize)
        self._busy_lock = threading.Lock()
        self._is_busy = False

    def _set_busy(self, value):
        with self._busy_lock:
            self._is_busy = value

    def is_busy(self):
        with self._busy_lock:
            return self._is_busy

    def consume_result(self):
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def _push_result(self, payload):
        _queue_put_latest(self._result_queue, payload)


class DecoderThread(_BaseWorkerThread):
    def __init__(self, fs, decoder_cfg, feature_cfg, model_bundle, name="DecoderThread", debug=False):
        super().__init__(name=name, input_maxsize=1, result_maxsize=1)
        self.fs = fs
        self.decoder_cfg = decoder_cfg
        self.feature_cfg = feature_cfg
        self.model_bundle = model_bundle
        self.debug = debug
        self._feature_plan = None
        self._feature_plan_channels = None
        self._stats = {
            'total_decodes': 0,
            'total_time_ms': 0.0,
            'max_preprocess_ms': 0.0,
            'max_decode_ms': 0.0,
        }

    def submit(self, decode_id, data, timestamp_s):
        payload = {
            'decode_id': int(decode_id),
            'timestamp_s': float(timestamp_s),
            'data': np.asarray(data, dtype=float),
            'queue_time': time.perf_counter(),
        }
        return _queue_put_latest(self._input_queue, payload)

    def decode_stop(self):
        self._stop_event.set()

    def _ensure_feature_plan(self, n_channels):
        if self._feature_plan is None or self._feature_plan_channels != n_channels:
            self._feature_plan = prepare_feature_plan(self.fs, self.feature_cfg, n_channels)
            self._feature_plan_channels = n_channels
        return self._feature_plan

    def _build_error_result(self, payload, error):
        return {
            'decode_id': payload.get('decode_id', -1),
            'data_received_time_s': round(payload.get('timestamp_s', 0.0), 6),
            'data_shape': tuple(payload.get('data', np.array([])).shape),
            'preprocess_time_ms': None,
            'feature_time_ms': None,
            'model_infer_time_ms': None,
            'result': {
                'predicted_target': None,
                'confidence': None,
                'probabilities': {},
                'success': False,
                'method': 'ml_decoder',
                'error': str(error),
            },
        }

    def _update_stats(self, preprocess_time_ms, decode_time_ms, total_pipeline_ms):
        self._stats['total_decodes'] += 1
        self._stats['total_time_ms'] += total_pipeline_ms
        self._stats['max_preprocess_ms'] = max(self._stats['max_preprocess_ms'], preprocess_time_ms)
        self._stats['max_decode_ms'] = max(self._stats['max_decode_ms'], decode_time_ms)

    def _decode_payload(self, payload):
        decode_id = payload['decode_id']
        data = payload['data']
        queue_wait_ms = (time.perf_counter() - payload['queue_time']) * 1000.0

        preprocess_start = time.perf_counter()
        preprocessed = preprocess_data(data, self.fs, self.decoder_cfg)
        preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

        feature_start = time.perf_counter()
        feature_plan = self._ensure_feature_plan(preprocessed.shape[-2])
        featured = extract_feature(preprocessed, self.fs, feature_plan=feature_plan)
        feature_time_ms = (time.perf_counter() - feature_start) * 1000.0

        model_infer_start = time.perf_counter()
        decode_result = decode(featured, self.model_bundle)
        model_infer_time_ms = (time.perf_counter() - model_infer_start) * 1000.0

        decode_time_ms = feature_time_ms + model_infer_time_ms
        total_pipeline_ms = preprocess_time_ms + decode_time_ms
        self._update_stats(preprocess_time_ms, decode_time_ms, total_pipeline_ms)

        if self.debug and total_pipeline_ms > 150:
            print(
                f'[Decoder-SLOW] Decode ID:{decode_id} | '
                f'Preprocess:{preprocess_time_ms:.1f}ms | '
                f'Feature:{feature_time_ms:.1f}ms | '
                f'ModelInfer:{model_infer_time_ms:.1f}ms | '
                f'Decode:{decode_time_ms:.1f}ms | '
                f'Total:{total_pipeline_ms:.1f}ms | '
                f'QueueWait:{queue_wait_ms:.1f}ms'
            )

        return {
            'decode_id': decode_id,
            'data_received_time_s': round(payload['timestamp_s'], 6),
            'data_shape': tuple(data.shape),
            'preprocess_time_ms': round(preprocess_time_ms, 3),
            'feature_time_ms': round(feature_time_ms, 3),
            'model_infer_time_ms': round(model_infer_time_ms, 3),
            'result': decode_result,
        }

    def run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._set_busy(True)
            try:
                result_payload = self._decode_payload(payload)
            except Exception as exc:
                result_payload = self._build_error_result(payload, exc)
            finally:
                self._set_busy(False)

            self._push_result(result_payload)

    def get_stats(self):
        total_decodes = self._stats['total_decodes']
        avg_time = self._stats['total_time_ms'] / total_decodes if total_decodes else 0.0
        return {
            'total_decodes': total_decodes,
            'avg_pipeline_ms': round(avg_time, 2),
            'max_preprocess_ms': round(self._stats['max_preprocess_ms'], 2),
            'max_decode_ms': round(self._stats['max_decode_ms'], 2),
        }


class StimThread(_BaseWorkerThread):
    def __init__(self, stimulator=None, name="StimThread", debug=False):
        super().__init__(name=name, input_maxsize=4, result_maxsize=20)
        self.stimulator = stimulator
        self.debug = debug
        self._stats = {
            'total_commands': 0,
            'start_commands': 0,
            'stop_commands': 0,
            'failed_commands': 0,
            'total_time_ms': 0.0,
            'max_command_time_ms': 0.0,
        }

    def submit(self, decode_id, command_type, params=None, command_label=""):
        payload = {
            'decode_id': int(decode_id),
            'command_type': str(command_type),
            'params': params,
            'command_label': command_label,
            'queue_time': time.perf_counter(),
        }
        try:
            self._input_queue.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def stim_stop(self):
        self._stop_event.set()
        try:
            if self.stimulator is not None:
                self.stimulator.stop_stimulation()
        except Exception:
            pass
        self._set_busy(False)

    def _build_result(self, payload, success, error="", queue_wait_ms=0.0, command_time_ms=0.0):
        return {
            'decode_id': payload['decode_id'],
            'command_type': payload['command_type'],
            'command_sent': int(bool(success)),
            'command_content': payload['command_label'],
            'error': error,
            'queue_wait_ms': round(queue_wait_ms, 2),
            'command_time_ms': round(command_time_ms, 2),
        }

    def _execute(self, payload):
        queue_wait_ms = (time.perf_counter() - payload['queue_time']) * 1000.0
        command_type = payload['command_type']
        params = payload['params']

        if self.stimulator is None or params is None:
            return self._build_result(payload, False, 'stimulator_unavailable', queue_wait_ms)

        command_start = time.perf_counter()
        if command_type == 'start':
            success = bool(self.stimulator.start_stimulation(params=params))
            self._stats['start_commands'] += 1
            error = '' if success else 'start_stim_failed'
        elif command_type == 'stop':
            success = bool(self.stimulator.stop_stimulation(channel=params.channel))
            self._stats['stop_commands'] += 1
            error = '' if success else 'stop_stim_failed'
        else:
            success = False
            error = f'unsupported_command:{command_type}'

        command_time_ms = (time.perf_counter() - command_start) * 1000.0
        if self.debug:
            print(
                f'[StimThread] Decode ID:{payload["decode_id"]} | '
                f'Type:{command_type} | Success:{int(success)} | '
                f'Command:{command_time_ms:.1f}ms | QueueWait:{queue_wait_ms:.1f}ms'
            )

        return self._build_result(payload, success, error, queue_wait_ms, command_time_ms)

    def _update_stats(self, command_time_ms, success):
        self._stats['total_commands'] += 1
        self._stats['total_time_ms'] += command_time_ms
        self._stats['max_command_time_ms'] = max(self._stats['max_command_time_ms'], command_time_ms)
        if not success:
            self._stats['failed_commands'] += 1

    def run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._set_busy(True)
            try:
                result_payload = self._execute(payload)
            except Exception as exc:
                result_payload = self._build_result(payload, False, str(exc))
            finally:
                self._set_busy(False)

            self._update_stats(result_payload['command_time_ms'], bool(result_payload['command_sent']))
            self._push_result(result_payload)

    def get_stats(self):
        total_commands = self._stats['total_commands']
        avg_time = self._stats['total_time_ms'] / total_commands if total_commands else 0.0
        return {
            'total_commands': total_commands,
            'start_commands': self._stats['start_commands'],
            'stop_commands': self._stats['stop_commands'],
            'failed_commands': self._stats['failed_commands'],
            'avg_command_ms': round(avg_time, 2),
            'max_command_ms': round(self._stats['max_command_time_ms'], 2),
        }
