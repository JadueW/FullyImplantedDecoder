# -*- coding: utf-8 -*-
import csv
import os
from datetime import datetime

import numpy as np
from psychopy import core


def monotonic_time_s():
    return float(core.monotonicClock.getTime())


def console_timestamp():
    return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]"


def print_timeline(tag, message, *, action_time_s=None):
    action_part = f" | t={action_time_s:.3f}s" if action_time_s is not None else ""
    print(f"{console_timestamp()} {tag}{action_part} | {message}")


def drain_thread_results(*workers):
    for worker in workers:
        while worker.consume_result() is not None:
            pass


def append_error_text(old_error, new_error):
    if not new_error:
        return old_error
    return f"{old_error};{new_error}" if old_error else new_error


def build_action_log_path(log_dir, participant_id, session_id):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(
        log_dir,
        f"action_decode_p{participant_id}_s{session_id}_{timestamp_str}.csv"
    )


def to_data_window(data, expected_channels):
    if (
        data is None
        or getattr(data, "size", 0) == 0
        or not hasattr(data, "ndim")
        or data.ndim != 2
    ):
        return np.empty((expected_channels, 0))
    return np.asarray(data, dtype=float)


def merge_new_chunk(window, new_chunk, window_points):
    if (
        new_chunk is None
        or getattr(new_chunk, "size", 0) == 0
        or not hasattr(new_chunk, "ndim")
        or new_chunk.ndim != 2
    ):
        return window, ""

    new_chunk = np.asarray(new_chunk, dtype=float)
    chunk_shape = str(tuple(new_chunk.shape))

    if window.size == 0:
        window = new_chunk
    elif new_chunk.shape[0] == window.shape[0]:
        window = np.concatenate([window, new_chunk], axis=1)
    else:
        print_timeline(
            "DECODE-WARN",
            f"channel mismatch | window={tuple(window.shape)} | new={tuple(new_chunk.shape)}",
        )
        return window, chunk_shape

    if window.ndim == 2 and window.shape[1] > window_points:
        window = window[:, -window_points:]

    return window, chunk_shape


def build_decode_log_row(exp_info, action_start_time_s, action_now_s, decode_payload, decode_result,
                         new_chunk_shape, total_pipeline_time_ms, should_stim, stim_submit_ok,
                         command_label, stim_error):
    predicted_target = decode_result.get("predicted_target", "")
    return {
        "decode_id": decode_payload.get("decode_id", -1),
        "participant": exp_info.get("participant", "unknown"),
        "session": exp_info.get("session", "001"),
        "action_start_time_s": round(action_start_time_s, 6),
        "action_end_time_s": "",
        "time_in_action_s": round(action_now_s - action_start_time_s, 6),
        "data_received_time_s": decode_payload.get("data_received_time_s", ""),
        "new_chunk_shape": new_chunk_shape,
        "data_shape": decode_payload.get("data_shape", ""),
        "downsample_shape": decode_payload.get("downsample_shape", ""),
        "processed_fs": decode_payload.get("processed_fs", ""),
        "preprocess_time_ms": decode_payload.get("preprocess_time_ms", ""),
        "feature_time_ms": decode_payload.get("feature_time_ms", ""),
        "model_infer_time_ms": decode_payload.get("model_infer_time_ms", ""),
        "total_pipeline_time_ms": total_pipeline_time_ms,
        "decode_result": predicted_target,
        "confidence": decode_result.get("confidence", ""),
        "should_stim": should_stim,
        "stim_submit_ok": stim_submit_ok,
        "command_sent": 0,
        "command_content": command_label,
        "stim_error": stim_error,
    }


def apply_stim_payload_to_logs(action_log_cache, action_pending_log_by_id, stim_payload):
    decode_id = stim_payload.get("decode_id")
    if decode_id not in action_pending_log_by_id:
        return

    row_idx = action_pending_log_by_id[decode_id]
    if not (0 <= row_idx < len(action_log_cache)):
        return

    row = action_log_cache[row_idx]
    if stim_payload.get("command_type") == "start":
        row["command_sent"] = stim_payload.get("command_sent", 0)

    returned_content = stim_payload.get("command_content", "")
    if returned_content:
        row["command_content"] = returned_content

    row["stim_error"] = append_error_text(row.get("stim_error", ""), stim_payload.get("error", ""))


def log_decode_event(decode_payload, decode_result, total_pipeline_time_ms, *, action_time_s):
    print_timeline(
        "DECODE",
        (
            f"id={decode_payload.get('decode_id')} | target={decode_result.get('predicted_target')} | "
            f"success={int(bool(decode_result.get('success', False)))} | "
            f"conf={decode_result.get('confidence', 0)} | "
            f"pre={decode_payload.get('preprocess_time_ms')}ms | "
            f"feat={decode_payload.get('feature_time_ms')}ms | "
            f"model={decode_payload.get('model_infer_time_ms')}ms | "
            f"total={total_pipeline_time_ms}ms"
        ),
        action_time_s=action_time_s,
    )


def log_stim_payload(stim_payload, *, action_time_s):
    print_timeline(
        "STIM-RESULT",
        (
            f"id={stim_payload.get('decode_id')} | type={stim_payload.get('command_type')} | "
            f"sent={stim_payload.get('command_sent', 0)} | "
            f"queue={stim_payload.get('queue_wait_ms', '')}ms | "
            f"command={stim_payload.get('command_time_ms', '')}ms | "
            f"content={stim_payload.get('command_content', '')} | "
            f"error={stim_payload.get('error', '') or 'none'}"
        ),
        action_time_s=action_time_s,
    )


def save_action_decode_logs(log_rows, csv_path):
    if not log_rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "decode_id",
        "participant",
        "session",
        "action_start_time_s",
        "action_end_time_s",
        "time_in_action_s",
        "data_received_time_s",
        "new_chunk_shape",
        "data_shape",
        "downsample_shape",
        "processed_fs",
        "preprocess_time_ms",
        "feature_time_ms",
        "model_infer_time_ms",
        "total_pipeline_time_ms",
        "decode_result",
        "confidence",
        "should_stim",
        "stim_submit_ok",
        "command_sent",
        "command_content",
        "stim_error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
