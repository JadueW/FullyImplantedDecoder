import threading
import time
import warnings
from copy import deepcopy

import numpy as np

from src.decoder.features_extract.feature_extract import extract_feature
from src.decoder.online_inference.ml_decoder.ml_decoder import decode
from src.decoder.preprocess.preprocessor import preprocess_data


class DataStreamer:

    def __init__(self, data_source, decoder_config, feature_config=None, model_bundle=None):
        self.data_source = data_source
        self.decoder_config = decoder_config
        self.feature_config = feature_config
        self.model_bundle = model_bundle

        self.fs = int(decoder_config.get('fs', 2000))
        self.num_ch = int(decoder_config.get('num_ch', 128))
        self.initial_window_s = float(decoder_config.get('preparation_window_size', 5.0))
        self.decode_interval_s = float(decoder_config.get('decode_interval', 0.1))
        self.decode_window_s = float(decoder_config.get('decode_window_size', self.initial_window_s))

        self.initial_window_points = int(self.initial_window_s * self.fs)
        self.decode_window_points = int(self.decode_window_s * self.fs)
        self.step_points = max(1, int(self.decode_interval_s * self.fs))

        self.is_streaming = False
        self.stream_start_time = 0.0
        self.session_start_time = 0.0

        self._decode_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.decode_window = None
        self.logs = []
        self._consumed_result_version = 0
        self._next_log_id = 1

        self.shared_state = {
            'result_version': 0,
            'latest_result': None,
            'latest_log': None,
            'latest_data_timestamp_s': 0.0,
            'latest_data_shape': (0, 0),
            'decode_window_shape': (0, 0),
            'is_decode_running': False,
            'decode_count': 0,
            'last_error': '',
        }

    def set_decoder(self, feature_config=None, model_bundle=None):
        if feature_config is not None:
            self.feature_config = feature_config
        if model_bundle is not None:
            self.model_bundle = model_bundle

    def start_streaming(self):
        with self._lock:
            if self.is_streaming:
                return True

            try:
                if hasattr(self.data_source, 'initialize_device') and hasattr(self.data_source, 'begin_collect'):
                    self.data_source.initialize_device(mode=0)
                    self.data_source.begin_collect()
                else:
                    if hasattr(self.data_source, 'connect'):
                        try:
                            self.data_source.connect()
                        except Exception:
                            pass
                    if hasattr(self.data_source, 'start'):
                        try:
                            self.data_source.start()
                        except Exception:
                            pass

                self.stream_start_time = time.time()
                self.is_streaming = True
                return True
            except Exception as exc:
                warnings.warn(f'Failed to start streaming: {exc}')
                return False

    def stop_streaming(self):
        self.stop_decode_session(wait=True)

        with self._lock:
            if not self.is_streaming:
                return True

            try:
                if hasattr(self.data_source, 'stop_collect'):
                    try:
                        self.data_source.stop_collect()
                    except Exception:
                        pass
                if hasattr(self.data_source, 'close_connection'):
                    try:
                        self.data_source.close_connection(stop_collect=False)
                    except TypeError:
                        self.data_source.close_connection()
                elif hasattr(self.data_source, 'close_conn'):
                    self.data_source.close_conn()

                self.is_streaming = False
                return True
            except Exception as exc:
                warnings.warn(f'Failed to stop streaming: {exc}')
                return False

    def _get_data(self, num_points):
        data = self.data_source.get_data(int(num_points))
        if data is None:
            return None
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.size == 0:
            return None
        return data

    def _relative_time_s(self):
        if self.stream_start_time <= 0:
            return 0.0
        return time.time() - self.stream_start_time

    def initialize_decode_window(self, duration_s=None, timeout_s=15.0):
        if duration_s is None:
            duration_s = self.initial_window_s

        num_points = int(duration_s * self.fs)
        deadline = time.perf_counter() + timeout_s

        while time.perf_counter() < deadline:
            data = self._get_data(num_points)
            if data is not None and data.shape[1] >= num_points:
                self.decode_window = data[:, -num_points:].copy()
                with self._lock:
                    self.shared_state['decode_window_shape'] = tuple(self.decode_window.shape)
                return self.decode_window.copy()
            time.sleep(0.05)

        raise RuntimeError('Failed to get initial decode window within timeout.')

    def flush_buffer(self):
        try:
            if hasattr(self.data_source, 'getBufferTime'):
                buffer_time_s = float(self.data_source.getBufferTime())
                buffer_points = max(0, int(buffer_time_s * self.fs))
                if buffer_points > 0:
                    self._get_data(buffer_points)
                    return buffer_points

            if hasattr(self.data_source, 'get_info'):
                info = self.data_source.get_info()
                buffer_points = int(info.get('num_buffer_sample', 0))
                if buffer_points > 0:
                    self._get_data(buffer_points)
                    return buffer_points
        except Exception as exc:
            warnings.warn(f'Failed to flush buffer: {exc}')

        return 0

    def _append_chunk(self, data_chunk):
        if self.decode_window is None:
            self.decode_window = data_chunk.copy()
            return self.decode_window.copy()

        chunk_points = data_chunk.shape[1]
        target_points = self.decode_window.shape[1]
        updated = np.concatenate([self.decode_window, data_chunk], axis=1)
        if updated.shape[1] > target_points:
            updated = updated[:, -target_points:]
        self.decode_window = updated
        return self.decode_window.copy()

    def _decode_loop(self):
        with self._lock:
            self.shared_state['is_decode_running'] = True
            self.shared_state['last_error'] = ''

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                data_chunk = self._get_data(self.step_points)
                if data_chunk is None or data_chunk.shape[1] < self.step_points:
                    self._stop_event.wait(0.01)
                    continue

                if data_chunk.shape[1] > self.step_points:
                    data_chunk = data_chunk[:, -self.step_points:]

                decode_window = self._append_chunk(data_chunk)
                data_timestamp_s = self._relative_time_s()

                preprocess_start = time.perf_counter()
                preprocessed = preprocess_data(decode_window, self.fs, self.decoder_config)
                preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

                decode_start = time.perf_counter()
                features = extract_feature(preprocessed, self.fs, self.feature_config)
                decode_result = decode(features, self.model_bundle)
                decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

                log_entry = {
                    'id': self._next_log_id,
                    'data_received_time_s': round(data_timestamp_s, 6),
                    'data_shape': tuple(data_chunk.shape),
                    'decode_window_shape': tuple(decode_window.shape),
                    'preprocess_time_ms': round(preprocess_time_ms, 3),
                    'decode_time_ms': round(decode_time_ms, 3),
                    'decode_result': decode_result.get('predicted_target'),
                    'confidence': decode_result.get('confidence'),
                    'command_sent': 0,
                    'command_content': '',
                }
                self._next_log_id += 1

                with self._lock:
                    self.logs.append(log_entry)
                    self.shared_state['result_version'] += 1
                    self.shared_state['latest_result'] = deepcopy(decode_result)
                    self.shared_state['latest_log'] = deepcopy(log_entry)
                    self.shared_state['latest_data_timestamp_s'] = data_timestamp_s
                    self.shared_state['latest_data_shape'] = tuple(data_chunk.shape)
                    self.shared_state['decode_window_shape'] = tuple(decode_window.shape)
                    self.shared_state['decode_count'] += 1
                    self.shared_state['last_error'] = ''

            except Exception as exc:
                with self._lock:
                    self.shared_state['last_error'] = str(exc)
                warnings.warn(f'Decode loop error: {exc}')

            elapsed = time.perf_counter() - loop_start
            sleep_left = self.decode_interval_s - elapsed
            if sleep_left > 0:
                self._stop_event.wait(sleep_left)

        with self._lock:
            self.shared_state['is_decode_running'] = False

    def start_decode_session(self, initial_window_s=None, timeout_s=15.0, flush_after_init=True):
        if self.feature_config is None or self.model_bundle is None:
            raise RuntimeError('feature_config and model_bundle must be set before decoding.')

        if not self.is_streaming:
            started = self.start_streaming()
            if not started:
                raise RuntimeError('Failed to start data streaming.')

        with self._lock:
            if self._decode_thread is not None and self._decode_thread.is_alive():
                warnings.warn('Decode session is already running.')
                return False

        self.session_start_time = time.time()
        self.logs = []
        self._next_log_id = 1
        self._consumed_result_version = 0

        self.initialize_decode_window(duration_s=initial_window_s, timeout_s=timeout_s)
        if flush_after_init:
            self.flush_buffer()

        self._stop_event.clear()
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            name='DataStreamerDecodeThread',
            daemon=True,
        )
        self._decode_thread.start()
        return True

    def stop_decode_session(self, wait=True):
        self._stop_event.set()

        thread = self._decode_thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        self._decode_thread = None
        return True

    def get_latest_result(self):
        with self._lock:
            result = self.shared_state['latest_result']
            return deepcopy(result) if result is not None else None

    def consume_latest_result(self):
        with self._lock:
            version = self.shared_state['result_version']
            if version <= self._consumed_result_version:
                return None

            self._consumed_result_version = version
            result = deepcopy(self.shared_state['latest_result'])
            log_entry = deepcopy(self.shared_state['latest_log'])

        return {
            'result': result,
            'log': log_entry,
            'version': version,
        }

    def update_latest_command(self, command_sent, command_content=''):
        with self._lock:
            if self.logs:
                self.logs[-1]['command_sent'] = int(command_sent)
                self.logs[-1]['command_content'] = command_content
            if self.shared_state['latest_log'] is not None:
                self.shared_state['latest_log']['command_sent'] = int(command_sent)
                self.shared_state['latest_log']['command_content'] = command_content

    def get_logs(self):
        with self._lock:
            return deepcopy(self.logs)

    def get_shared_state(self):
        with self._lock:
            return deepcopy(self.shared_state)

    def get_decode_window(self):
        if self.decode_window is None:
            return None
        return self.decode_window.copy()

    def __del__(self):
        try:
            self.stop_streaming()
        except Exception:
            pass
