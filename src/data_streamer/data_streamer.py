import queue
import threading
import time
import gc
from datetime import datetime

import numpy as np

from src.decoder.features_extract.feature_extract import extract_feature, prepare_feature_plan
from src.decoder.online_inference.ml_decoder.ml_decoder import decode
from src.decoder.preprocess.preprocessor import preprocess_data


class DecoderThread(threading.Thread):
    # Decoder thread: preprocesses data, extracts features, and runs decoding.
    def __init__(self, fs, decoder_cfg, feature_cfg, model_bundle, name="DecoderThread", debug=False):
        super().__init__(name=name, daemon=True)
        self.fs = fs
        self.decoder_cfg = decoder_cfg
        self.feature_cfg = feature_cfg
        self.model_bundle = model_bundle
        self.debug = debug
        self._feature_plan = None
        self._feature_plan_channels = None

        self._stop_event = threading.Event()
        self._input_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._busy_lock = threading.Lock()
        self._is_busy = False

        # 性能统计
        self._stats = {
            'total_decodes': 0,
            'total_time_ms': 0.0,
            'max_preprocess_ms': 0.0,
            'max_decode_ms': 0.0,
        }

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
            "queue_time": time.perf_counter(),  # 记录入队时间
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
                queue_time = payload["queue_time"]

                # 计算队列等待时间
                queue_wait_ms = (time.perf_counter() - queue_time) * 1000.0

                # 预处理
                preprocess_start = time.perf_counter()
                preprocessed = preprocess_data(data, self.fs, self.decoder_cfg)
                preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

                # 特征提取 + 解码
                decode_start = time.perf_counter()
                n_channels = preprocessed.shape[-2]
                if self._feature_plan is None or self._feature_plan_channels != n_channels:
                    self._feature_plan = prepare_feature_plan(self.fs, self.feature_cfg, n_channels)
                    self._feature_plan_channels = n_channels

                featured = extract_feature(
                    preprocessed,
                    self.fs,
                    feature_plan=self._feature_plan,
                )
                decode_result = decode(featured, self.model_bundle)
                decode_time_ms = (time.perf_counter() - decode_start) * 1000.0

                total_pipeline_ms = preprocess_time_ms + decode_time_ms

                # 更新统计
                self._stats['total_decodes'] += 1
                self._stats['total_time_ms'] += total_pipeline_ms
                self._stats['max_preprocess_ms'] = max(self._stats['max_preprocess_ms'], preprocess_time_ms)
                self._stats['max_decode_ms'] = max(self._stats['max_decode_ms'], decode_time_ms)

                # 性能监控：如果处理时间超过150ms，打印详细信息
                if self.debug and total_pipeline_ms > 150:
                    gc_count = gc.get_count()
                    gc_thresholds = gc.get_threshold()
                    print(f'[Decoder-SLOW] Decode ID:{decode_id} | '
                          f'Preprocess:{preprocess_time_ms:.1f}ms | '
                          f'Decode:{decode_time_ms:.1f}ms | '
                          f'Total:{total_pipeline_ms:.1f}ms | '
                          f'QueueWait:{queue_wait_ms:.1f}ms | '
                          f'GC:{gc_count} | Thresholds:{gc_thresholds}')

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

    def get_stats(self):
        """返回解码线程的统计信息"""
        if self._stats['total_decodes'] > 0:
            avg_time = self._stats['total_time_ms'] / self._stats['total_decodes']
        else:
            avg_time = 0.0

        return {
            'total_decodes': self._stats['total_decodes'],
            'avg_pipeline_ms': round(avg_time, 2),
            'max_preprocess_ms': round(self._stats['max_preprocess_ms'], 2),
            'max_decode_ms': round(self._stats['max_decode_ms'], 2),
        }


class StimThread(threading.Thread):
    """Execute short stimulator I/O commands without owning timing policy."""

    def __init__(self, stimulator=None, name="StimThread", debug=False):
        super().__init__(name=name, daemon=True)
        self.stimulator = stimulator
        self.debug = debug

        self._stop_event = threading.Event()
        self._command_queue = queue.Queue(maxsize=4)
        self._result_queue = queue.Queue(maxsize=20)
        self._busy_lock = threading.Lock()
        self._is_busy = False

        # 性能统计
        self._stats = {
            'total_commands': 0,
            'start_commands': 0,
            'stop_commands': 0,
            'failed_commands': 0,
            'total_time_ms': 0.0,
            'max_command_time_ms': 0.0,
        }

    def _set_busy(self, value):
        with self._busy_lock:
            self._is_busy = value

    def is_busy(self):
        with self._busy_lock:
            return self._is_busy

    def _push_result(self, result_payload):
        while True:
            try:
                self._result_queue.put_nowait(result_payload)
                return
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    return

    def submit(self, decode_id, command_type, params=None, command_label=""):
        payload = {
            "decode_id": int(decode_id),
            "command_type": str(command_type),
            "params": params,
            "command_label": command_label,
            "queue_time": time.perf_counter(),  # 记录入队时间
        }

        try:
            self._command_queue.put_nowait(payload)
            return True
        except queue.Full:
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
        self._set_busy(False)

    def get_stats(self):
        """返回刺激线程的统计信息"""
        if self._stats['total_commands'] > 0:
            avg_time = self._stats['total_time_ms'] / self._stats['total_commands']
        else:
            avg_time = 0.0

        return {
            'total_commands': self._stats['total_commands'],
            'start_commands': self._stats['start_commands'],
            'stop_commands': self._stats['stop_commands'],
            'failed_commands': self._stats['failed_commands'],
            'avg_command_ms': round(avg_time, 2),
            'max_command_ms': round(self._stats['max_command_time_ms'], 2),
        }

    def run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._command_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            decode_id = payload["decode_id"]
            command_type = payload["command_type"]
            params = payload["params"]
            command_label = payload["command_label"]
            queue_time = payload["queue_time"]

            # 计算队列等待时间
            queue_wait_ms = (time.perf_counter() - queue_time) * 1000.0

            if self.stimulator is None or params is None:
                error_reason = "stimulator未连接" if self.stimulator is None else "参数为空"
                timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                print(f'{timestamp_str} ❌ stimulator {command_type} 失败 | decode_id:{decode_id} | '
                      f'错误:{error_reason} | 命令:{command_label}')

                self._push_result(
                    {
                        "decode_id": decode_id,
                        "command_type": command_type,
                        "command_sent": 0,
                        "command_content": command_label,
                        "error": "stimulator_unavailable",
                        "queue_wait_ms": round(queue_wait_ms, 2),
                    }
                )
                continue

            success = 0
            error_message = ""
            command_start = time.perf_counter()

            self._set_busy(True)
            try:
                if command_type == "start":
                    # 设置参数（同步设置，确保参数生效）
                    set_params_start = time.perf_counter()
                    params_ok = bool(self.stimulator.set_stimulation_params(params))
                    set_params_time = (time.perf_counter() - set_params_start) * 1000.0

                    if not params_ok:
                        error_message = "set_params_failed"
                        # 打印参数设置失败信息
                        timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                        print(f'{timestamp_str} ❌ stimulator start 失败 | decode_id:{decode_id} | '
                              f'错误:set_params_failed | 耗时:{set_params_time:.1f}ms')
                    else:
                        # 启动刺激
                        start_start = time.perf_counter()
                        success = int(bool(self.stimulator.start_stimulation(channel=params.channel)))
                        start_time = (time.perf_counter() - start_start) * 1000.0
                        total_time = (time.perf_counter() - command_start) * 1000.0

                        if not success:
                            error_message = "start_stim_failed"
                            # 打印启动失败信息
                            timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                            print(f'{timestamp_str} ❌ stimulator start 失败 | decode_id:{decode_id} | '
                                  f'错误:start_stim_failed | 参数设置:{set_params_time:.1f}ms | '
                                  f'启动尝试:{start_time:.1f}ms | 总耗时:{total_time:.1f}ms')
                        else:
                            # 打印启动成功信息
                            timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                            channel_info = params.channel if hasattr(params, 'channel') else 'AB'
                            print(f'{timestamp_str} ✅ stimulator start 成功 | decode_id:{decode_id} | '
                                  f'通道:{channel_info} | 参数设置:{set_params_time:.1f}ms | '
                                  f'启动刺激:{start_time:.1f}ms | 总耗时:{total_time:.1f}ms | '
                                  f'队列等待:{queue_wait_ms:.1f}ms')

                        if self.debug:
                            print(f'[StimThread-调试] Start命令详情 | Decode ID:{decode_id} | '
                                  f'参数设置:{set_params_time:.1f}ms | '
                                  f'启动刺激:{start_time:.1f}ms | '
                                  f'总计:{total_time:.1f}ms | '
                                  f'队列等待:{queue_wait_ms:.1f}ms')

                        self._stats['start_commands'] += 1

                elif command_type == "stop":
                    stop_start = time.perf_counter()
                    success = int(bool(self.stimulator.stop_stimulation(channel=params.channel)))
                    stop_time = (time.perf_counter() - stop_start) * 1000.0
                    total_time = (time.perf_counter() - command_start) * 1000.0

                    if not success:
                        error_message = "stop_stim_failed"
                        # 打印停止失败信息
                        timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                        print(f'{timestamp_str} ❌ stimulator stop 失败 | decode_id:{decode_id} | '
                              f'错误:stop_stim_failed | 停止尝试:{stop_time:.1f}ms | 总耗时:{total_time:.1f}ms')
                    else:
                        # 打印停止成功信息
                        timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                        channel_info = params.channel if hasattr(params, 'channel') else 'AB'
                        print(f'{timestamp_str} ✅ stimulator stop 成功 | decode_id:{decode_id} | '
                              f'通道:{channel_info} | 停止刺激:{stop_time:.1f}ms | 总耗时:{total_time:.1f}ms | '
                              f'队列等待:{queue_wait_ms:.1f}ms')

                    if self.debug:
                        print(f'[StimThread-调试] Stop命令详情 | Decode ID:{decode_id} | '
                              f'Stop:{stop_time:.1f}ms | '
                              f'Total:{total_time:.1f}ms | '
                              f'QueueWait:{queue_wait_ms:.1f}ms')

                    self._stats['stop_commands'] += 1
                else:
                    error_message = f"unsupported_command:{command_type}"
                    # 打印不支持的命令信息
                    timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                    print(f'{timestamp_str} ❌ stimulator 不支持命令 | decode_id:{decode_id} | '
                          f'命令类型:{command_type} | 错误:{error_message}')

            except Exception as e:
                error_message = str(e)
                success = 0
                total_time = (time.perf_counter() - command_start) * 1000.0

                # 打印异常信息
                timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                print(f'{timestamp_str} ❌ stimulator {command_type} 异常 | decode_id:{decode_id} | '
                      f'错误:{e} | 耗时:{total_time:.1f}ms')

                if self.debug:
                    import traceback
                    print(f'[StimThread-调试] 异常详情 | Decode ID:{decode_id} | '
                          f'Type:{command_type} | '
                          f'Error:{e} | '
                          f'Time:{total_time:.1f}ms')
                    traceback.print_exc()
            finally:
                self._set_busy(False)

            # 更新统计
            total_time = (time.perf_counter() - command_start) * 1000.0
            self._stats['total_commands'] += 1
            self._stats['total_time_ms'] += total_time
            self._stats['max_command_time_ms'] = max(self._stats['max_command_time_ms'], total_time)
            if success == 0:
                self._stats['failed_commands'] += 1

            # 性能监控：如果命令执行时间超过500ms，打印警告
            if total_time > 500:
                gc_count = gc.get_count()
                gc_thresholds = gc.get_threshold()
                print(f'[StimThread-SLOW] Decode ID:{decode_id} | Type:{command_type} | '
                      f'Time:{total_time:.0f}ms | QueueWait:{queue_wait_ms:.1f}ms | '
                      f'GC:{gc_count} | Thresholds:{gc_thresholds} | Failed:{success==0}')

            self._push_result(
                {
                    "decode_id": decode_id,
                    "command_type": command_type,
                    "command_sent": success,
                    "command_content": command_label,
                    "error": error_message,
                    "queue_wait_ms": round(queue_wait_ms, 2),
                }
            )
