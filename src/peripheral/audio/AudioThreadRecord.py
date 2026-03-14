import threading
import time
import os
import sys
import json
import wave
import pyaudio

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


def read_audio_config(device="audio", config_file_path=None):
    """
        device: audio / experiment
        """
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"读取{device}配置文件成功: {config.get(device, {})}")
        return config.get(device, {})
    except Exception as e:
        print(f"读取{device}配置文件失败")
        # 返回默认配置
        exit(0)


class AudioConfig:
    # 读取配置文件
    config_file = os.path.join( r"config\upper_limb_movement_config.json")
    # 获取实验配置
    audio_config = read_audio_config(device="audio", config_file_path=config_file)

    # 从配置中获取各项设置
    save_dir = audio_config.get("save_dir", "recordings")
    channels = audio_config.get("channels", 1)  # 单声道
    rate = audio_config.get("rate", 44100)  # 采样率
    chunk = audio_config.get("chunk", 1024)  # 缓冲区大小
    format = audio_config.get("format", 8)  # 采样位数 (8表示pyaudio.paInt16)
    device_index = audio_config.get("device_index", None)  # 麦克风设备索引


class AudioThread(threading.Thread):
    def __init__(self, name, exit_event):
        super(AudioThread, self).__init__()
        self.name = name
        self.exit_event = exit_event
        self.p = pyaudio.PyAudio()

        # 如果没有指定设备索引，则列出所有可用设备并使用默认设备
        if AudioConfig.device_index is None:
            print("可用的音频输入设备:")
            for i in range(self.p.get_device_count()):
                dev_info = self.p.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels') > 0:  # 只显示输入设备
                    print(f"{i}: {dev_info.get('name')}")
            # 使用默认输入设备
            info = self.p.get_default_input_device_info()
            AudioConfig.device_index = info['index']
            print(f"使用默认输入设备: {info['name']} (索引: {AudioConfig.device_index})")

        # 确保audio文件夹存在
        save_dir = os.path.join(os.getcwd(), AudioConfig.save_dir, "audio_recordings")
        print(f"save_dir: {save_dir}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # if not os.path.exists("./audio"): 
        #     os.makedirs("./audio")

        # 创建录音文件
        # self.filename = f"./audio/{int(time.time())}_audio.wav"
        self.filename = os.path.join(save_dir, f"{int(time.time())}_audio.wav")
        self.wf = wave.open(self.filename, 'wb')
        self.wf.setnchannels(AudioConfig.channels)
        self.wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        self.wf.setframerate(AudioConfig.rate)

        # 打开音频流
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=AudioConfig.channels,
            rate=AudioConfig.rate,
            input=True,
            input_device_index=AudioConfig.device_index,
            frames_per_buffer=AudioConfig.chunk
        )
        print(f"音频录制初始化完成，将保存到: {self.filename}")

    def run(self):
        print(f"开始录制音频: {self.name}")

        while not self.exit_event.is_set():
            # 读取音频数据并写入文件
            try:
                data = self.stream.read(AudioConfig.chunk, exception_on_overflow=False)
                self.wf.writeframes(data)
            except (IOError, OSError):
                # 当流被关闭时会发生此错误，这在退出时是正常的
                if self.exit_event.is_set():
                    break
            except Exception as e:
                print(f"音频录制错误: {e}")

        # 退出清理
        self.cleanup()

    def cleanup(self):
        # 停止并关闭音频流
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # 关闭音频文件
        if hasattr(self, 'wf') and self.wf:
            self.wf.close()

        # 终止PyAudio
        if hasattr(self, 'p') and self.p:
            self.p.terminate()

        print(f"音频录制已停止，文件已保存: {self.filename}")


audio_exit_event = threading.Event()


def init_audio_thread():
    global audio_exit_event
    audio_thread = AudioThread("AudioRecordThread", audio_exit_event)
    return audio_thread


def start_audio_thread(audio_thread, mode="production"):
    # 开始线程

    audio_thread.start()

    if mode == "development":
        # 线程阻塞
        audio_thread.join()
    print("Audio recording started.")


def exit_audio_thread(audio_thread):
    print("准备结束音频录制。Ready to Exit Audio Record. Time: ", time.time())
    global audio_exit_event
    if not audio_exit_event.is_set():
        # 通知线程停止
        print("Setting audio exit event...")
        audio_exit_event.set()

    if audio_thread and audio_thread.is_alive():
        print("音频录制线程正在关闭...")
        audio_thread.join()  # 使用join()等待线程完成
    print("音频录制已结束。Audio recording exit. Time: ", time.time())


if __name__ == '__main__':
    audio_thread = init_audio_thread()
    start_audio_thread(audio_thread, mode="development")
