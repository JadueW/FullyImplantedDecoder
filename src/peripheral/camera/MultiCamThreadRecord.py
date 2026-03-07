import threading
import time
import os
import json

import peripheral.mvsdk as mvsdk


def read_config(device="camera",config_file_path=None):
    """
    device: camera / experiment
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


class Config:
# class CameraConfig:
    # 读取配置文件
    config_file = r"D:\dev\paradigm_dev\config\upper_limb_movement_config.json"
    # 获取实验配置
    camera_config = read_config(device="camera",config_file_path=config_file)

    # 从配置中获取各项设置
    exposure_time = camera_config.get("exposure_time", 10)
    rgbgain = camera_config.get("rgbgain", [1, 1, 1])
    video_quality = camera_config.get("video_quality", 70)
    cam_gamma = camera_config.get("cam_gamma", 1.0)
    cam_Contrast = camera_config.get("cam_Contrast", 100)
    cam_Saturation = camera_config.get("cam_Saturation", 100)
    cam_Sharpness = camera_config.get("cam_Sharpness", 0)
    cam_AnalogGain = camera_config.get("cam_AnalogGain", 0)
    save_dir = camera_config.get("save_dir", "recordings")
    # cam_Brightness = camera_config.get("cam_Brightness", 0)

    # # 设置保存路径
    # save_dir = os.path.join(os.path.expanduser('~'), 'Desktop', save_dir)

    save_dir = os.path.join(os.getcwd(), save_dir, "video_recordings")
    print(f"save_dir: {save_dir}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



class CameraThread(threading.Thread):
    def __init__(self, n, camera_index, exit_event):
        super(CameraThread, self).__init__()
        self.n = n
        self.camera_index = camera_index
        self.exit_event = exit_event

        # 枚举相机
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        if camera_index >= nDev:
            print(f"Invalid camera index {camera_index}. There are only {nDev} cameras.")
            return

        DevInfo = DevList[camera_index]
        print(f"Using camera {camera_index}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        except mvsdk.CameraException as e:
            print(f"CameraInit Failed({e.error_code}): {e.message}")
            return

        cap = mvsdk.CameraGetCapability(self.hCamera)

        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, int(Config.exposure_time) * 1000)
        # 设置其他相机参数
        mvsdk.CameraSetGain(self.hCamera, int(Config.rgbgain[0]), int(Config.rgbgain[1]), int(Config.rgbgain[2]))
        mvsdk.CameraSetGamma(self.hCamera, int(Config.cam_gamma))
        mvsdk.CameraSetContrast(self.hCamera, Config.cam_Contrast)
        mvsdk.CameraSetSaturation(self.hCamera, Config.cam_Saturation)
        mvsdk.CameraSetSharpness(self.hCamera, Config.cam_Sharpness)
        mvsdk.CameraSetAnalogGain(self.hCamera, Config.cam_AnalogGain)
        mvsdk.CameraSetRotate(self.hCamera, 2)

        mvsdk.CameraPlay(self.hCamera)

        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        # 这边注意有没有F盘，另外如果失败可以将第二个参数4改成0，代表不压缩，看看是否可以录像，如果可以代表电脑h264编码器有点问题，需要自己处理下
        self.ErrCode = mvsdk.CameraInitRecord(
            self.hCamera, 4,
            # f"./recordings/video_recordings/{int(time.time())}_video{self.camera_index}_q70-h264.mp4",
            os.path.join(Config.save_dir, f"{int(time.time())}_video{self.camera_index}_q70-h264.mp4"),
            True, 70, 25
        )
        print(f"CameraInitRecord={self.ErrCode}")

    def run(self):
        if self.ErrCode != 0:
            print("初始化录像失败的。")
            return

        print(f"Camera {self.camera_index}: {self.hCamera}")

        while not self.exit_event.is_set():
            try:
                # mvsdk.CameraSoftTrigger(self.hCamera)
                # 将超时时间从2000ms缩短到200ms，可以更快地响应退出事件
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                if pRawData:
                    mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                    mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                    mvsdk.CameraPushFrame(self.hCamera, self.pFrameBuffer, FrameHead)
            except mvsdk.CameraException as e:
                print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")

        # 退出清理
        self.cleanup()

    def cleanup(self):
        # 异常退出线程后
        mvsdk.CameraStopRecord(self.hCamera)
        mvsdk.CameraUnInit(self.hCamera)
        mvsdk.CameraAlignFree(self.pFrameBuffer)
        print(f"Camera {self.camera_index} stopped and resources cleaned up.")


exit_event = threading.Event()


def init_camera_thread():
    global exit_event
    # 枚举并显示所有可用相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No cameras found!")
        exit()

    print("Available cameras:")
    for i, DevInfo in enumerate(DevList):
        print(f"{i}: {DevInfo.GetFriendlyName()}")

    selected_indices = [i for i in range(nDev)]

    threads = []

    # 初始化录像
    for index in selected_indices:
        if index < nDev:
            t = CameraThread(f"CameraThread-{index}", index, exit_event)
            threads.append(t)
    return threads


def start_camera_thread(threads, mode="production"):
    # 开始线程
    for t in threads:
        t.start()

    if mode == "development":
        # 线程阻塞
        for t in threads:
            t.join()
    print("开始录制视频，Camera recording started.")



def exit_camera_thread(threads):
    print("准备结束视频录制。时间: ", time.time())
    global exit_event
    if not exit_event.is_set():
        # 通知各线程停止
        print("Setting camera exit event...")
        exit_event.set()

    # 等待所有线程结束
    for t in threads:
        if t and t.is_alive():
            t.join()  # 使用join()等待线程完成
    print("所有摄像机线程已结束。时间: ", time.time())


if __name__ == '__main__':
    t_list = init_camera_thread()
    start_camera_thread(t_list, mode="development")
    pass
