"""
解码日志记录

对应需求：
解码日志的记录，包括但是不限于：
   - 接收到数据时间
   - 数据shape
   - 解码耗时
   - 解码结果
   - 是否输出刺激指令
   - 指令内容
   - 单独一个文件输出
"""

import sys
import os
import time
import json
import csv
import warnings
from datetime import datetime
from pathlib import Path
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DecoderLogger:

    def __init__(self, output_dir, config):
        """
        初始化解码日志记录
        :param output_dir: 日志输出目录
        :param config: 配置字典，包含日志相关参数
        """

        self.output_dir = Path(output_dir)
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳作为日志文件标识
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_files = {
            'csv': self.output_dir / f"decoder_log_{timestamp}.csv",
            'json': self.output_dir / f"decoder_log_{timestamp}.jsonl",
            'txt': self.output_dir / f"decoder_log_{timestamp}.txt",
            'summary': self.output_dir / f"decoder_summary_{timestamp}.json"
        }

        self.csv_columns = [
            # 时间信息
            'timestamp',
            'absolute_time',
            'trial_id',

            # 任务信息
            'action_name',         # 动作名称
            'action_type',         # 动作类型
            'hand',                # 左右手
            'expected_target',     # 预期目标

            # 数据信息
            'data_shape',          # 数据shape
            'data_channels',       # 通道数
            'data_samples',        # 样本数
            'data_received_time',  # 接收到数据的时间

            # 解码信息
            'decode_count',        # 解码计数（trial内第几次）
            'decode_start_time',   # 解码开始时间
            'decode_end_time',     # 解码结束时间
            'decode_duration',     # 解码耗时（秒）

            # 解码结果
            'decode_success',      # 解码是否成功
            'predicted_target',    # 预测目标
            'confidence',          # 置信度
            'all_probabilities',   # 所有类别的概率
            'decode_method',       # 解码方法

            # 刺激器指令
            'command_sent',        # 是否输出刺激指令
            'command_type',        # 指令类型
            'command_details',     # 指令内容
            'stimulator_channel',  # 刺激器通道
            'stimulator_duration', # 刺激器持续时间

            # 其他信息
            'buffer_status',       # 缓冲区状态
            'error_message',       # 错误信息
            'notes'                # 备注信息
        ]

        self.stats = {
            'total_decode_attempts': 0,     # 总解码尝试次数
            'successful_decodes': 0,        # 成功解码次数
            'failed_decodes': 0,            # 失败解码次数
            'total_commands_sent': 0,       # 总发送指令数
            'total_decode_time': 0.0,       # 总解码时间
            'avg_decode_time': 0.0,         # 平均解码时间
            'max_decode_time': 0.0,         # 最大解码时间
            'min_decode_time': float('inf'),# 最小解码时间
            'decode_results': {},           # 各类别的解码次数统计
            'command_by_target': {}         # 各目标收到的指令次数
        }

        self.is_closed = False
        self._lock = threading.Lock()
        self._decode_count = 0

        self._initialize_files()

        print(f"\n=== 解码日志记录器初始化完成 ===")
        print(f"日志目录: {self.output_dir}")
        print(f"CSV日志: {self.log_files['csv'].name}")
        print(f"JSON日志: {self.log_files['json'].name}")
        print(f"文本日志: {self.log_files['txt'].name}")

    def _initialize_files(self):
        """
        初始化日志文件
        """
        try:
            # 1. 初始化CSV文件
            with open(self.log_files['csv'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

            # 2. 初始化文本日志文件
            with open(self.log_files['txt'], 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("解码日志记录\n")
                f.write("=" * 80 + "\n")
                f.write(f"开始时间: {datetime.now().isoformat()}\n")
                f.write(f"配置信息: {json.dumps(self.config, ensure_ascii=False, indent=2)}\n")
                f.write("=" * 80 + "\n\n")

            print("日志文件初始化完成")

        except Exception as e:
            print(f"初始化日志文件失败: {e}")
            raise

    def log(self,timestamp,trial_id,data_shape,decode_time,decode_result,command_sent,command_details, **kwargs):
        """
        记录一次解码过程
        :param timestamp: 相对于实验开始的时间戳
        :param trial_id: Trial编号
        :param data_shape: 数据的shape (n_channels, n_timepoints)
        :param decode_time: 解码耗时
        :param decode_result: 解码结果
        :param command_sent: 是否发送了刺激指令
        :param command_details: 指令详情
        :param kwargs:
        :return:
        """
        if self.is_closed:
            warnings.warn("日志记录器已关闭，无法记录新日志")
            return False

        try:
            with self._lock:
                # 增加解码计数
                self._decode_count += 1
                self.stats['total_decode_attempts'] += 1

                log_entry = {
                    # 时间信息
                    'timestamp': round(timestamp, 6),
                    'absolute_time': datetime.now().isoformat(),
                    'trial_id': trial_id,

                    # 任务信息（从kwargs获取）
                    'action_name': kwargs.get('action_name', ''),
                    'action_type': kwargs.get('action_type', ''),
                    'hand': kwargs.get('hand', ''),
                    'expected_target': kwargs.get('expected_target', ''),

                    # 数据信息
                    'data_shape': str(data_shape),
                    'data_channels': data_shape[0] if len(data_shape) > 0 else 0,
                    'data_samples': data_shape[1] if len(data_shape) > 1 else 0,
                    'data_received_time': timestamp,

                    # 解码计数
                    'decode_count': self._decode_count,
                    'decode_start_time': kwargs.get('decode_start_time', timestamp - decode_time),
                    'decode_end_time': timestamp,

                    # 解码耗时
                    'decode_duration': round(decode_time, 6),

                    # 解码结果
                    'decode_success': False,
                    'predicted_target': '',
                    'confidence': 0.0,
                    'all_probabilities': '',
                    'decode_method': '',

                    # 刺激器指令
                    'command_sent': command_sent,
                    'command_type': '',
                    'command_details': command_details if command_details else '',
                    'stimulator_channel': kwargs.get('stimulator_channel', ''),
                    'stimulator_duration': kwargs.get('stimulator_duration', ''),

                    # 其他信息
                    'buffer_status': json.dumps(kwargs.get('buffer_status', {}), ensure_ascii=False),
                    'error_message': kwargs.get('error_message', ''),
                    'notes': kwargs.get('notes', '')
                }

                if decode_result:
                    log_entry['decode_success'] = decode_result.get('success', False)
                    log_entry['predicted_target'] = decode_result.get('predicted_target', '')
                    log_entry['confidence'] = decode_result.get('confidence', 0.0)
                    log_entry['all_probabilities'] = json.dumps(
                        decode_result.get('probabilities', {}),
                        ensure_ascii=False
                    )
                    log_entry['decode_method'] = decode_result.get('method', '')

                    # 更新统计信息
                    if decode_result.get('success', False):
                        self.stats['successful_decodes'] += 1
                        predicted = decode_result.get('predicted_target', 'unknown')
                        self.stats['decode_results'][predicted] = \
                            self.stats['decode_results'].get(predicted, 0) + 1
                    else:
                        self.stats['failed_decodes'] += 1

                if command_sent and command_details:
                    try:
                        if isinstance(command_details, str):
                            cmd_dict = json.loads(command_details)
                        else:
                            cmd_dict = command_details
                        log_entry['command_type'] = cmd_dict.get('type', '')
                        log_entry['stimulator_channel'] = cmd_dict.get('channel', '')
                        log_entry['stimulator_duration'] = cmd_dict.get('duration', '')
                    except:
                        pass

                    # 更新统计
                    self.stats['total_commands_sent'] += 1
                    target = log_entry['predicted_target'] or 'unknown'
                    self.stats['command_by_target'][target] = \
                        self.stats['command_by_target'].get(target, 0) + 1

                self.stats['total_decode_time'] += decode_time
                self.stats['avg_decode_time'] = \
                    self.stats['total_decode_time'] / self.stats['total_decode_attempts']
                self.stats['max_decode_time'] = \
                    max(self.stats['max_decode_time'], decode_time)
                self.stats['min_decode_time'] = \
                    min(self.stats['min_decode_time'], decode_time)

                self._write_to_files(log_entry)

                return True

        except Exception as e:
            print(f"记录解码日志失败: {e}")
            return False

    def _write_to_files(self, log_entry):

        try:
            with open(self.log_files['csv'], 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writerow(log_entry)

            with open(self.log_files['json'], 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            with open(self.log_files['txt'], 'a', encoding='utf-8') as f:
                f.write("-" * 80 + "\n")
                f.write(f"解码 #{log_entry['decode_count']} | Trial: {log_entry['trial_id']} | "
                       f"时间: {log_entry['timestamp']:.3f}s\n")
                f.write(f"  动作: {log_entry['action_name']} ({log_entry['action_type']}) | "
                       f"手: {log_entry['hand']}\n")
                f.write(f"  数据: {log_entry['data_shape']} | "
                       f"耗时: {log_entry['decode_duration']*1000:.2f}ms\n")
                f.write(f"  解码: {'成功' if log_entry['decode_success'] else '失败'} | "
                       f"预测: {log_entry['predicted_target']} | "
                       f"置信度: {log_entry['confidence']:.3f}\n")
                f.write(f"  刺激: {'是' if log_entry['command_sent'] else '否'} | "
                       f"指令: {log_entry['command_details']}\n")
                if log_entry['error_message']:
                    f.write(f"  错误: {log_entry['error_message']}\n")
                f.write("\n")

        except Exception as e:
            print(f"写入日志文件失败: {e}")

    def log_trial_start(self, trial_id, trial_info):
        try:
            with open(self.log_files['txt'], 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Trial {trial_id} 开始\n")
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"信息: {json.dumps(trial_info, ensure_ascii=False, indent=2)}\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            print(f"记录Trial开始失败: {e}")

    def log_trial_end(self, trial_id, trial_stats):
        try:
            with open(self.log_files['txt'], 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Trial {trial_id} 结束\n")
                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"统计: {json.dumps(trial_stats, ensure_ascii=False, indent=2)}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"记录Trial结束失败: {e}")

    def get_stats(self):

        return self.stats.copy()

    def print_summary(self):

        print("\n" + "=" * 60)
        print("解码日志汇总")
        print("=" * 60)

        stats = self.stats
        total = stats['total_decode_attempts']

        if total > 0:
            print(f"\n总解码次数: {total}")
            print(f"成功: {stats['successful_decodes']} ({stats['successful_decodes']/total*100:.1f}%)")
            print(f"失败: {stats['failed_decodes']} ({stats['failed_decodes']/total*100:.1f}%)")
            print(f"发送指令: {stats['total_commands_sent']}")
            print(f"\n解码耗时:")
            print(f"  平均: {stats['avg_decode_time']*1000:.2f} ms")
            print(f"  最大: {stats['max_decode_time']*1000:.2f} ms")
            print(f"  最小: {stats['min_decode_time']*1000:.2f} ms")

            if stats['decode_results']:
                print(f"\n解码结果分布:")
                for target, count in sorted(stats['decode_results'].items()):
                    print(f"  {target}: {count} 次")

            if stats['command_by_target']:
                print(f"\n指令发送分布:")
                for target, count in sorted(stats['command_by_target'].items()):
                    print(f"  {target}: {count} 次")

        print("=" * 60 + "\n")

    def save_summary(self):

        try:
            summary = {
                'end_time': datetime.now().isoformat(),
                'stats': self.stats,
                'log_files': {k: str(v) for k, v in self.log_files.items()}
            }

            with open(self.log_files['summary'], 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"汇总统计已保存到: {self.log_files['summary']}")

        except Exception as e:
            print(f"保存汇总统计失败: {e}")

    def close(self):

        if self.is_closed:
            return

        try:
            with self._lock:
                self.print_summary()

                self.save_summary()

                with open(self.log_files['txt'], 'a', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("日志记录结束\n")
                    f.write(f"结束时间: {datetime.now().isoformat()}\n")
                    f.write("=" * 80 + "\n")

                self.is_closed = True
                print("解码日志记录器已关闭")

        except Exception as e:
            print(f"关闭日志记录器失败: {e}")

    def __del__(self):

        if not self.is_closed:
            try:
                self.close()
            except:
                pass

