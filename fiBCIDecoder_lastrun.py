# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.5),
    on 三月 17, 2026, at 11:56
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins

plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from init_code
import sys
import os
import json
import warnings
import csv
from datetime import datetime
import time
from psychopy import logging, core

# 1. 导入项目模块
from src.transmission.trans_control.dataController import DataController
from src.peripheral.camera.MultiCamThreadRecord import (
    init_camera_thread,
    start_camera_thread,
    exit_camera_thread
)
from src.peripheral.audio.AudioThreadRecord import (
    init_audio_thread,
    start_audio_thread,
    exit_audio_thread
)

from src.data_streamer.data_streamer import DecoderThread, StimThread
from src.decoder.online_inference.ml_decoder.ml_decoder import load_model
from src.welink_stimulator.stimulator_controller import (
    ChannelStimConfig,
    StimulatorController,
    StimulationParams,
)

from src.utils.exit_handler import patch_core_quit, register_cleanup_function


# 公共函数
def monotonic_time_s():
    return float(core.monotonicClock.getTime())


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
        "preprocess_time_ms",
        "decode_time_ms",
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


# 2. 读取配置文件
config_path = os.path.join('./config', 'upper_limb_movement_config.json')

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info(f"✓ Config loaded from {config_path}")
except Exception as e:
    logging.error(f"✗ Failed to load config: {e}")
    config = {}
    warnings.warn("Using default configuration values")

with open(os.path.join("./config", "decoder_config.json"), "r", encoding="utf-8") as f:
    decoder_cfg = json.load(f)

with open(os.path.join("./config", "feature_config.json"), "r", encoding="utf-8") as f:
    feature_cfg = json.load(f)

model_path = os.path.join("./pretrained_models", "ml_models", "fine_decoder.pkl")
model_bundle = load_model(model_path)
print(f"模型加载成功，model = {model_bundle['model']}")

# 3. 获取配置参数
host = config.get("experiment", {}).get("host", "127.0.0.1")
port = config.get("experiment", {}).get("port", 5001)

big_text_size = config.get("experiment", {}).get("big_text_size", 0.25)
small_text_size = config.get("experiment", {}).get("small_text_size", 0.25)

trial_repeats = config.get("experiment", {}).get("trial_repetas", 1)
block_repeats = config.get("experiment", {}).get("block_repeats", 1)

task_type = config.get("experiment", {}).get("task_type", 0)
task_desc = config.get("experiment", {}).get("task_desc", {"0": "粗大", "1": "精细"})

if task_type == 1:
    selected_rows = "4:7"
    task_num = 3
else:
    selected_rows = "0:4"
    task_num = 4
    if task_type != 0:
        print("WARNING: 任务类型只能为0或1，默认为0.")

conditions_file = config.get(
    "experiment", {}
).get(
    "conditions",
    r".\resources\conditions\upper_limb_movement\condition.csv"
)

block_conditions_file = config.get(
    "experiment", {}
).get(
    "block_conditions",
    r".\resources\conditions\upper_limb_movement\block_condition.csv"
)

sound_dir = config.get(
    "experiment", {}
).get(
    "sound_dir",
    r".\resources\audios\upper_limb_movement"
)

video_dir = config.get(
    "experiment", {}
).get(
    "video_dir",
    r".\resources\videos\upperlimb"
)

video_direction = config.get("experiment", {}).get("video_direction", "right")
video_dir = os.path.join(video_dir, video_direction)

random_seed = config.get("experiment", {}).get("random_seed", 42)

logging.info("Experiment configuration:")
logging.info(f"  Host: {host}")
logging.info(f"  Task type: {task_type} ({task_desc.get(str(task_type), 'unknown')})")
logging.info(f"  Block repeats: {block_repeats}")
logging.info(f"  Trial repeats: {trial_repeats}")

camera_connected_status = config.get(
    "experiment", {}
).get(
    "peripheral_stats", {}
).get(
    "is_camera_connected", False
)

audio_connected_status = config.get(
    "experiment", {}
).get(
    "peripheral_stats", {}
).get(
    "is_audio_connected", False
)

# 4. 初始化 DataController
dc = None
try:
    dc = DataController(host, port, 4, 2000)
    dc.connect()
    dc.start()
    dc.setBufferSize(10)
    logging.info("✓ DataController initialized and started")
except Exception as e:
    logging.error(f"✗ DataController initialization failed: {e}")
    warnings.warn(f"Could not initialize DataController: {e}")
    print(f"ERROR: {e}")

# 5. 初始化相机录制
camera_threads = None
if camera_connected_status:
    try:
        camera_threads = init_camera_thread()
        start_camera_thread(camera_threads, mode="production")
        logging.info("✓ Camera recording started")
    except Exception as e:
        logging.error(f"✗ Camera initialization failed: {e}")
        warnings.warn(f"Could not start camera recording: {e}")

# 6. 初始化音频录制
audio_thread = None
if audio_connected_status:
    try:
        audio_thread = init_audio_thread()
        start_audio_thread(audio_thread, mode="production")
        logging.info("✓ Audio recording started")
    except Exception as e:
        logging.error(f"✗ Audio initialization failed: {e}")
        warnings.warn(f"Could not start audio recording: {e}")

# 7. 初始化刺激器
stimulator = None
stimulator_enabled = False
try:
    with open(os.path.join("config", "upper_limb_movement_config.json"), "r", encoding="utf-8") as f:
        stim_cfg = json.load(f)

    stim_port = stim_cfg.get("experiment", {}).get("stim_com_port", None)

    if stim_port:
        stimulator = StimulatorController(
            port=stim_port,
            baudrate=230400,  # 优化方案2: 提高波特率从115200到230400
            timeout=0.1,      # 减少超时从1.0到0.1秒
            debug=True,       # 启用调试模式，查看详细通信信息
        )
        stimulator_enabled = stimulator.connect()
except Exception as e:
    print(f"Stimulator init failed: {e}")
    stimulator = None
    stimulator_enabled = False

# 8. 初始化 decoder 线程
decoder_thread = DecoderThread(
    fs=int(decoder_cfg["fs"]),
    decoder_cfg=decoder_cfg,
    feature_cfg=feature_cfg,
    model_bundle=model_bundle,
)

# 9. 初始化stim线程
stim_thread = StimThread(
    stimulator=stimulator if stimulator_enabled else None,
)


# 10. 注册清理函数
def cleanup_all_resources():
    logging.info("=" * 50)
    logging.info("Cleaning up resources...")

    if dc:
        try:
            dc.close_conn()
            logging.info("✓ DataController closed")
        except Exception as e:
            logging.error(f"✗ Error closing DataController: {e}")

    if audio_thread:
        try:
            exit_audio_thread(audio_thread)
            logging.info("✓ Audio recording stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping audio: {e}")

    if camera_threads:
        try:
            exit_camera_thread(camera_threads)
            logging.info("✓ Camera recording stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping cameras: {e}")

    if decoder_thread:
        try:
            decoder_thread.decode_stop()
            logging.info("✓ decoder_thread stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping decoder_thread: {e}")

    if stimulator:
        try:
            stimulator.disconnect()
            logging.info("✓ stimulator stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping stimulator: {e}")

    if stim_thread:
        try:
            stim_thread.stim_stop()
            logging.info("✓ stim_thread stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping stim_thread: {e}")

    logging.info("=" * 50)
    logging.info("All resources cleaned up")


register_cleanup_function(cleanup_all_resources)
patch_core_quit()

import random
import numpy as np

random.seed(random_seed)
np.random.seed(random_seed)

logging.info("=" * 50)
logging.info("Upper Limb Movement Paradigm Initialized")
logging.info(f"Task: {task_desc.get(str(task_type), 'unknown')}")
logging.info("=" * 50)

# 11. 解码类别 -> 刺激参数映射
# route: 通路，A通道 0-7 对应 A0-A7；B通道 0-7 对应 B0-B7
# prescription_id: 处方号，1-13
# level: 档位，1-32
# duration_code: 1=10min, 2=20min, 3=30min
stim_param_map = {
    0: StimulationParams(
        channel="A",
        a=ChannelStimConfig(route=0, prescription_id=12, level=1, duration_code=1),
    ),
    1: StimulationParams(
        channel="B",
        b=ChannelStimConfig(route=0, prescription_id=2, level=1, duration_code=1),
    ),
}

# 12. 全局变量
action_decode_log_dir = os.path.join("./logs", "action_decode_logs")
os.makedirs(action_decode_log_dir, exist_ok=True)

action_start_time_s = 0.0
action_end_time_s = 0.0
action_latest_payload = None
action_latest_result = None
action_latest_log = None
action_command_sent = 0
action_command_text = ""
action_decode_count = 0
action_log_cache = []
action_log_file_path = ""

decode_interval_s = 0.1
decode_window_s = 5.0
stim_duration_ms = int(config.get("experiment", {}).get("stim_duration_ms", 500))
window_points = int(decode_window_s * decoder_cfg["fs"])
global_decode_id = 1

stim_window_active = False
stim_window_decode_id = None
stim_window_row_idx = None
stim_window_command_label = ""
stim_window_params = None
stim_window_deadline_s = 0.0

stim_stop_pending = False
stim_stop_decode_id = None

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.5'
expName = 'fiBCIDecoder'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2048, 1280]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.

    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.

    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)

    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)

    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\dev\\FullyImplantedDecoder\\fiBCIDecoder_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.

    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.

    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename + '.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )

    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window

    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.

    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')

    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1, -1, -1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='+')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()

    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to
    the device manager (deviceManager)

    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}

    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')

    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)

    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)

    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer

    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # create speaker 'br_sound'
    deviceManager.addDevice(
        deviceName='br_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'tr_sound'
    deviceManager.addDevice(
        deviceName='tr_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'act_sound'
    deviceManager.addDevice(
        deviceName='act_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'to_sound'
    deviceManager.addDevice(
        deviceName='to_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'bo_sound'
    deviceManager.addDevice(
        deviceName='bo_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True


def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.

    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return

    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.

    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess

    # Start Code - component code to be run after the window creation

    # --- Initialize components for Routine "init" ---
    # Run 'Begin Experiment' code from init_code
    # 启动两个线程
    decoder_thread.start()
    stim_thread.start()
    print("decoder_thread启动成功！")
    print("stim_thread启动成功！")
    init_text = visual.TextStim(win=win, name='init_text',
                                text='+',
                                font='Arial',
                                pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                                color='white', colorSpace='rgb', opacity=None,
                                languageStyle='LTR',
                                depth=-1.0);

    # --- Initialize components for Routine "block_ready" ---
    br_text = visual.TextStim(win=win, name='br_text',
                              text='',
                              font='Arial',
                              pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                              color='white', colorSpace='rgb', opacity=None,
                              languageStyle='LTR',
                              depth=-1.0);
    br_3 = visual.TextStim(win=win, name='br_3',
                           text='3',
                           font='Arial',
                           pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                           color='white', colorSpace='rgb', opacity=None,
                           languageStyle='LTR',
                           depth=-2.0);
    br_2 = visual.TextStim(win=win, name='br_2',
                           text='2',
                           font='Arial',
                           pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                           color='white', colorSpace='rgb', opacity=None,
                           languageStyle='LTR',
                           depth=-3.0);
    br_1 = visual.TextStim(win=win, name='br_1',
                           text='1',
                           font='Arial',
                           pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                           color='white', colorSpace='rgb', opacity=None,
                           languageStyle='LTR',
                           depth=-4.0);
    br_sound = sound.Sound(
        'A',
        secs=-1,
        stereo=True,
        hamming=True,
        speaker='br_sound', name='br_sound'
    )
    br_sound.setVolume(1.0)

    # --- Initialize components for Routine "trial_ready" ---
    tr_text = visual.TextStim(win=win, name='tr_text',
                              text='',
                              font='Arial',
                              pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                              color='white', colorSpace='rgb', opacity=None,
                              languageStyle='LTR',
                              depth=-1.0);
    tr_sound = sound.Sound(
        'A',
        secs=-1,
        stereo=True,
        hamming=True,
        speaker='tr_sound', name='tr_sound'
    )
    tr_sound.setVolume(1.0)

    # --- Initialize components for Routine "action" ---
    act_movie = visual.MovieStim(
        win, name='act_movie',
        filename=None, movieLib='ffpyplayer',
        loop=True, volume=1.0, noAudio=True,
        pos=(0, 0), size=(1.2, 0.675), units=win.units,
        ori=0.0, anchor='center', opacity=None, contrast=1.0,
        depth=-1
    )
    act_sound = sound.Sound(
        'A',
        secs=-1,
        stereo=True,
        hamming=True,
        speaker='act_sound', name='act_sound'
    )
    act_sound.setVolume(1.0)

    # --- Initialize components for Routine "trial_over" ---
    to_text = visual.TextStim(win=win, name='to_text',
                              text='',
                              font='Arial',
                              pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                              color='white', colorSpace='rgb', opacity=None,
                              languageStyle='LTR',
                              depth=-1.0);
    to_sound = sound.Sound(
        'A',
        secs=-1,
        stereo=True,
        hamming=True,
        speaker='to_sound', name='to_sound'
    )
    to_sound.setVolume(1.0)

    # --- Initialize components for Routine "inter_trial_interval" ---
    iti_text = visual.TextStim(win=win, name='iti_text',
                               text='随机休息',
                               font='Arial',
                               pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0,
                               color='white', colorSpace='rgb', opacity=None,
                               languageStyle='LTR',
                               depth=-1.0);

    # --- Initialize components for Routine "block_over" ---
    bo_text = visual.TextStim(win=win, name='bo_text',
                              text='',
                              font='Arial',
                              pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                              color='white', colorSpace='rgb', opacity=None,
                              languageStyle='LTR',
                              depth=-1.0);
    bo_sound = sound.Sound(
        'A',
        secs=-1,
        stereo=True,
        hamming=True,
        speaker='bo_sound', name='bo_sound'
    )
    bo_sound.setVolume(1.0)

    # --- Initialize components for Routine "finish" ---
    finish_text = visual.TextStim(win=win, name='finish_text',
                                  text='全部结束',
                                  font='Arial',
                                  pos=(0, 0), draggable=False, height=0.25, wrapWidth=None, ori=0.0,
                                  color='white', colorSpace='rgb', opacity=None,
                                  languageStyle='LTR',
                                  depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')

    # create some handy timers

    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )

    # --- Prepare to start Routine "init" ---
    # create an object to store info about Routine init
    init = data.Routine(
        name='init',
        components=[init_text],
    )
    init.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for init
    init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    init.tStart = globalClock.getTime(format='float')
    init.status = STARTED
    thisExp.addData('init.started', init.tStart)
    init.maxDuration = None
    # keep track of which components have finished
    initComponents = init.components
    for thisComponent in init.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "init" ---
    init.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *init_text* updates

        # if init_text is starting this frame...
        if init_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            init_text.frameNStart = frameN  # exact frame index
            init_text.tStart = t  # local t and not account for scr refresh
            init_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(init_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'init_text.started')
            # update status
            init_text.status = STARTED
            init_text.setAutoDraw(True)

        # if init_text is active this frame...
        if init_text.status == STARTED:
            # update params
            pass

        # if init_text is stopping this frame...
        if init_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > init_text.tStartRefresh + 3 - frameTolerance:
                # keep track of stop time/frame for later
                init_text.tStop = t  # not accounting for scr refresh
                init_text.tStopRefresh = tThisFlipGlobal  # on global time
                init_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'init_text.stopped')
                # update status
                init_text.status = FINISHED
                init_text.setAutoDraw(False)

        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp,
                win=win,
                timers=[routineTimer],
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            init.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in init.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "init" ---
    for thisComponent in init.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for init
    init.tStop = globalClock.getTime(format='float')
    init.tStopRefresh = tThisFlipGlobal
    thisExp.addData('init.stopped', init.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if init.maxDurationReached:
        routineTimer.addTime(-init.maxDuration)
    elif init.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()

    # set up handler to look after randomisation of conditions etc
    block_loop = data.TrialHandler2(
        name='block_loop',
        nReps=block_repeats,
        method='random',
        extraInfo=expInfo,
        originPath=-1,
        trialList=data.importConditions(
            block_conditions_file,
            selection=selected_rows
        )
        ,
        seed=None,
    )
    thisExp.addLoop(block_loop)  # add the loop to the experiment
    thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop:
            globals()[paramName] = thisBlock_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()

    for thisBlock_loop in block_loop:
        currentLoop = block_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
        if thisBlock_loop != None:
            for paramName in thisBlock_loop:
                globals()[paramName] = thisBlock_loop[paramName]

        # --- Prepare to start Routine "block_ready" ---
        # create an object to store info about Routine block_ready
        block_ready = data.Routine(
            name='block_ready',
            components=[br_text, br_3, br_2, br_1, br_sound],
        )
        block_ready.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from br_code
        br_sound_dir = os.path.join(sound_dir, block_audio_name)
        br_text.setText(block_stimuli)
        br_sound.setSound(br_sound_dir, secs=8, hamming=True)
        br_sound.setVolume(1.0, log=False)
        br_sound.seek(0)
        # store start times for block_ready
        block_ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_ready.tStart = globalClock.getTime(format='float')
        block_ready.status = STARTED
        thisExp.addData('block_ready.started', block_ready.tStart)
        block_ready.maxDuration = None
        # keep track of which components have finished
        block_readyComponents = block_ready.components
        for thisComponent in block_ready.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "block_ready" ---
        # if trial has changed, end Routine now
        if isinstance(block_loop, data.TrialHandler2) and thisBlock_loop.thisN != block_loop.thisTrial.thisN:
            continueRoutine = False
        block_ready.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 8.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            # *br_text* updates

            # if br_text is starting this frame...
            if br_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                # keep track of start time/frame for later
                br_text.frameNStart = frameN  # exact frame index
                br_text.tStart = t  # local t and not account for scr refresh
                br_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(br_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'br_text.started')
                # update status
                br_text.status = STARTED
                br_text.setAutoDraw(True)

            # if br_text is active this frame...
            if br_text.status == STARTED:
                # update params
                pass

            # if br_text is stopping this frame...
            if br_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > br_text.tStartRefresh + 5 - frameTolerance:
                    # keep track of stop time/frame for later
                    br_text.tStop = t  # not accounting for scr refresh
                    br_text.tStopRefresh = tThisFlipGlobal  # on global time
                    br_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'br_text.stopped')
                    # update status
                    br_text.status = FINISHED
                    br_text.setAutoDraw(False)

            # *br_3* updates

            # if br_3 is starting this frame...
            if br_3.status == NOT_STARTED and tThisFlip >= 5 - frameTolerance:
                # keep track of start time/frame for later
                br_3.frameNStart = frameN  # exact frame index
                br_3.tStart = t  # local t and not account for scr refresh
                br_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(br_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'br_3.started')
                # update status
                br_3.status = STARTED
                br_3.setAutoDraw(True)

            # if br_3 is active this frame...
            if br_3.status == STARTED:
                # update params
                pass

            # if br_3 is stopping this frame...
            if br_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > br_3.tStartRefresh + 1.0 - frameTolerance:
                    # keep track of stop time/frame for later
                    br_3.tStop = t  # not accounting for scr refresh
                    br_3.tStopRefresh = tThisFlipGlobal  # on global time
                    br_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'br_3.stopped')
                    # update status
                    br_3.status = FINISHED
                    br_3.setAutoDraw(False)

            # *br_2* updates

            # if br_2 is starting this frame...
            if br_2.status == NOT_STARTED and tThisFlip >= 6 - frameTolerance:
                # keep track of start time/frame for later
                br_2.frameNStart = frameN  # exact frame index
                br_2.tStart = t  # local t and not account for scr refresh
                br_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(br_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'br_2.started')
                # update status
                br_2.status = STARTED
                br_2.setAutoDraw(True)

            # if br_2 is active this frame...
            if br_2.status == STARTED:
                # update params
                pass

            # if br_2 is stopping this frame...
            if br_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > br_2.tStartRefresh + 1.0 - frameTolerance:
                    # keep track of stop time/frame for later
                    br_2.tStop = t  # not accounting for scr refresh
                    br_2.tStopRefresh = tThisFlipGlobal  # on global time
                    br_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'br_2.stopped')
                    # update status
                    br_2.status = FINISHED
                    br_2.setAutoDraw(False)

            # *br_1* updates

            # if br_1 is starting this frame...
            if br_1.status == NOT_STARTED and tThisFlip >= 7 - frameTolerance:
                # keep track of start time/frame for later
                br_1.frameNStart = frameN  # exact frame index
                br_1.tStart = t  # local t and not account for scr refresh
                br_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(br_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'br_1.started')
                # update status
                br_1.status = STARTED
                br_1.setAutoDraw(True)

            # if br_1 is active this frame...
            if br_1.status == STARTED:
                # update params
                pass

            # if br_1 is stopping this frame...
            if br_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > br_1.tStartRefresh + 1.0 - frameTolerance:
                    # keep track of stop time/frame for later
                    br_1.tStop = t  # not accounting for scr refresh
                    br_1.tStopRefresh = tThisFlipGlobal  # on global time
                    br_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'br_1.stopped')
                    # update status
                    br_1.status = FINISHED
                    br_1.setAutoDraw(False)

            # *br_sound* updates

            # if br_sound is starting this frame...
            if br_sound.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                # keep track of start time/frame for later
                br_sound.frameNStart = frameN  # exact frame index
                br_sound.tStart = t  # local t and not account for scr refresh
                br_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('br_sound.started', tThisFlipGlobal)
                # update status
                br_sound.status = STARTED
                br_sound.play(when=win)  # sync with win flip

            # if br_sound is stopping this frame...
            if br_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > br_sound.tStartRefresh + 8 - frameTolerance or br_sound.isFinished:
                    # keep track of stop time/frame for later
                    br_sound.tStop = t  # not accounting for scr refresh
                    br_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    br_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'br_sound.stopped')
                    # update status
                    br_sound.status = FINISHED
                    br_sound.stop()

            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[br_sound]
                )
                # skip the frame we paused on
                continue

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_ready.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_ready.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # --- Ending Routine "block_ready" ---
        for thisComponent in block_ready.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_ready
        block_ready.tStop = globalClock.getTime(format='float')
        block_ready.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_ready.stopped', block_ready.tStop)
        br_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if block_ready.maxDurationReached:
            routineTimer.addTime(-block_ready.maxDuration)
        elif block_ready.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-8.000000)

        # set up handler to look after randomisation of conditions etc
        trial_loop = data.TrialHandler2(
            name='trial_loop',
            nReps=trial_repeats,
            method='random',
            extraInfo=expInfo,
            originPath=-1,
            trialList=data.importConditions(
                conditions_file,
                selection=selected_rows
            )
            ,
            seed=None,
        )
        thisExp.addLoop(trial_loop)  # add the loop to the experiment
        thisTrial_loop = trial_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
        if thisTrial_loop != None:
            for paramName in thisTrial_loop:
                globals()[paramName] = thisTrial_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()

        for thisTrial_loop in trial_loop:
            currentLoop = trial_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_loop.rgb)
            if thisTrial_loop != None:
                for paramName in thisTrial_loop:
                    globals()[paramName] = thisTrial_loop[paramName]

            # --- Prepare to start Routine "trial_ready" ---
            # create an object to store info about Routine trial_ready
            trial_ready = data.Routine(
                name='trial_ready',
                components=[tr_text, tr_sound],
            )
            trial_ready.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from tr_code
            tr_tex = stimuli
            tr_sound_dir = os.path.join(sound_dir, audio_name)
            tr_text.setText(tr_tex)
            tr_sound.setSound(tr_sound_dir, secs=3, hamming=True)
            tr_sound.setVolume(1.0, log=False)
            tr_sound.seek(0)
            # store start times for trial_ready
            trial_ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_ready.tStart = globalClock.getTime(format='float')
            trial_ready.status = STARTED
            thisExp.addData('trial_ready.started', trial_ready.tStart)
            trial_ready.maxDuration = None
            # keep track of which components have finished
            trial_readyComponents = trial_ready.components
            for thisComponent in trial_ready.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "trial_ready" ---
            # if trial has changed, end Routine now
            if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
                continueRoutine = False
            trial_ready.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *tr_text* updates

                # if tr_text is starting this frame...
                if tr_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    tr_text.frameNStart = frameN  # exact frame index
                    tr_text.tStart = t  # local t and not account for scr refresh
                    tr_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(tr_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'tr_text.started')
                    # update status
                    tr_text.status = STARTED
                    tr_text.setAutoDraw(True)

                # if tr_text is active this frame...
                if tr_text.status == STARTED:
                    # update params
                    pass

                # if tr_text is stopping this frame...
                if tr_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > tr_text.tStartRefresh + 3 - frameTolerance:
                        # keep track of stop time/frame for later
                        tr_text.tStop = t  # not accounting for scr refresh
                        tr_text.tStopRefresh = tThisFlipGlobal  # on global time
                        tr_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'tr_text.stopped')
                        # update status
                        tr_text.status = FINISHED
                        tr_text.setAutoDraw(False)

                # *tr_sound* updates

                # if tr_sound is starting this frame...
                if tr_sound.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    tr_sound.frameNStart = frameN  # exact frame index
                    tr_sound.tStart = t  # local t and not account for scr refresh
                    tr_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('tr_sound.started', tThisFlipGlobal)
                    # update status
                    tr_sound.status = STARTED
                    tr_sound.play(when=win)  # sync with win flip

                # if tr_sound is stopping this frame...
                if tr_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > tr_sound.tStartRefresh + 3 - frameTolerance or tr_sound.isFinished:
                        # keep track of stop time/frame for later
                        tr_sound.tStop = t  # not accounting for scr refresh
                        tr_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        tr_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'tr_sound.stopped')
                        # update status
                        tr_sound.status = FINISHED
                        tr_sound.stop()

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[tr_sound]
                    )
                    # skip the frame we paused on
                    continue

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_ready.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_ready.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "trial_ready" ---
            for thisComponent in trial_ready.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_ready
            trial_ready.tStop = globalClock.getTime(format='float')
            trial_ready.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_ready.stopped', trial_ready.tStop)
            tr_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_ready.maxDurationReached:
                routineTimer.addTime(-trial_ready.maxDuration)
            elif trial_ready.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)

            # --- Prepare to start Routine "action" ---
            # create an object to store info about Routine action
            action = data.Routine(
                name='action',
                components=[act_movie, act_sound],
            )
            action.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from act_code
            act_movie_dir = os.path.join(video_dir, video_name)
            act_sound_dir = os.path.join(sound_dir, "开始.wav")
            global global_decode_id

            action_start_time_s = monotonic_time_s()
            action_end_time_s = 0.0
            action_now_s = 0.0
            action_last_fetch_s = action_start_time_s

            action_decode_count = 0
            action_log_cache = []
            action_pending_chunk_shape_by_id = {}
            action_pending_log_by_id = {}

            while decoder_thread.consume_result() is not None:
                pass

            while stim_thread.consume_result() is not None:
                pass

            if stimulator_enabled and stimulator is not None and getattr(stimulator, "is_stimulating", False):
                try:
                    stimulator.stop_stimulation()
                except Exception as e:
                    print(f"WARNING: stop residual stimulation failed at action begin: {e}")

            stim_window_active = False
            stim_window_decode_id = None
            stim_window_row_idx = None
            stim_window_command_label = ""
            stim_window_params = None
            stim_window_deadline_s = 0.0

            stim_stop_pending = False
            stim_stop_decode_id = None

            fs = int(decoder_cfg["fs"])
            step_points = int(decode_interval_s * fs)
            window_points = int(decode_window_s * fs)
            expected_channels = 128

            participant_id = expInfo.get("participant", "unknown")
            session_id = expInfo.get("session", "001")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            action_log_file_path = os.path.join(
                action_decode_log_dir,
                f"action_decode_p{participant_id}_s{session_id}_{timestamp_str}.csv"
            )

            action_decode_window = dc.get_data(window_points)

            if (
                    action_decode_window is None
                    or getattr(action_decode_window, "size", 0) == 0
                    or not hasattr(action_decode_window, "ndim")
                    or action_decode_window.ndim != 2
            ):
                action_decode_window = np.empty((expected_channels, 0))
            else:
                action_decode_window = np.asarray(action_decode_window, dtype=float)

            print(f"initial_decode_window.shape = {action_decode_window.shape}")

            act_movie.setMovie(act_movie_dir)
            act_sound.setSound(act_sound_dir, secs=3, hamming=True)
            act_sound.setVolume(1.0, log=False)
            act_sound.seek(0)

            act_movie.setMovie(act_movie_dir)
            act_sound.setSound(act_sound_dir, secs=3, hamming=True)
            act_sound.setVolume(1.0, log=False)
            act_sound.seek(0)
            # store start times for action
            action.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            action.tStart = globalClock.getTime(format='float')
            action.status = STARTED
            thisExp.addData('action.started', action.tStart)
            action.maxDuration = None
            # keep track of which components have finished
            actionComponents = action.components
            for thisComponent in action.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "action" ---
            # if trial has changed, end Routine now
            if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
                continueRoutine = False
            action.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from act_code
                action_now_s = monotonic_time_s()

                # 1. 到了 500ms 截止时间，就结束业务窗口，并异步提交 stop
                if stim_window_active and action_now_s >= stim_window_deadline_s:
                    stim_window_active = False

                    timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                    start_time = stim_window_deadline_s - stim_duration_ms/1000.0
                    print(f'{timestamp_str} ⏰ 刺激窗口截止时间到达 | decode_id:{stim_window_decode_id} | '
                          f'窗口:{start_time:.3f}s - {stim_window_deadline_s:.3f}s | '
                          f'持续:{stim_duration_ms}ms | 当前:{action_now_s:.3f}s')

                    if (
                            (not stim_stop_pending)
                            and stim_window_params is not None
                            and (not stim_thread.is_busy())
                    ):
                        stop_submit_ok = stim_thread.submit(
                            decode_id=stim_window_decode_id,
                            command_type="stop",
                            params=stim_window_params,
                            command_label=stim_window_command_label,
                        )

                        if stop_submit_ok:
                            stim_stop_pending = True
                            stim_stop_decode_id = stim_window_decode_id
                            print(f"[{action_now_s:.3f}s] 刺激停止命令已发送 | Decode ID: {stim_window_decode_id}")
                        else:
                            print(f"[{action_now_s:.3f}s] 刺激停止命令发送失败 | 线程忙或队列满")

                # 2. 每 100ms 固定取一次 100ms 数据块
                if (action_now_s - action_last_fetch_s) >= decode_interval_s:
                    new_chunk = dc.get_data(step_points)
                    current_new_chunk_shape = ""
                    action_last_fetch_s = action_now_s

                    if (
                            new_chunk is not None
                            and getattr(new_chunk, "size", 0) > 0
                            and hasattr(new_chunk, "ndim")
                            and new_chunk.ndim == 2
                    ):
                        new_chunk = np.asarray(new_chunk, dtype=float)
                        current_new_chunk_shape = str(tuple(new_chunk.shape))

                        if action_decode_window.size == 0:
                            action_decode_window = new_chunk
                        elif new_chunk.shape[0] == action_decode_window.shape[0]:
                            action_decode_window = np.concatenate(
                                [action_decode_window, new_chunk],
                                axis=1
                            )
                        else:
                            print(
                                f"WARNING: new_chunk channel mismatch, "
                                f"window={action_decode_window.shape}, new={new_chunk.shape}"
                            )

                        if (
                                action_decode_window.ndim == 2
                                and action_decode_window.shape[1] > window_points
                        ):
                            action_decode_window = action_decode_window[:, -window_points:]

                        if (
                                action_decode_window.ndim == 2
                                and action_decode_window.shape[1] >= window_points
                                and not decoder_thread.is_busy()
                        ):
                            decode_id = global_decode_id
                            decode_input = action_decode_window[:, -window_points:]

                            submit_ok = decoder_thread.submit(
                                decode_id=decode_id,
                                data=decode_input,
                                timestamp_s=action_now_s,
                            )

                            if submit_ok:
                                global_decode_id += 1
                                action_pending_chunk_shape_by_id[decode_id] = current_new_chunk_shape

                # 3. 解码完成后返回结果
                decode_payload = decoder_thread.consume_result()

                if decode_payload is not None:
                    action_decode_count += 1
                    decode_id = decode_payload["decode_id"]
                    decode_result = decode_payload["result"]

                    predicted_target = decode_result.get("predicted_target", None)
                    decode_success = bool(decode_result.get("success", False))

                    preprocess_time_ms = decode_payload.get("preprocess_time_ms", "")
                    decode_time_ms = decode_payload.get("decode_time_ms", "")

                    total_pipeline_time_ms = ""
                    if preprocess_time_ms is not None and decode_time_ms is not None:
                        total_pipeline_time_ms = round(
                            float(preprocess_time_ms) + float(decode_time_ms),
                            3
                        )

                    new_chunk_shape = action_pending_chunk_shape_by_id.pop(decode_id, "")

                    want_stim = decode_success and (predicted_target == 1 or predicted_target == 0)
                    should_stim = int(want_stim)
                    stim_submit_ok = 0
                    command_sent = 0
                    command_label = ""
                    stim_error = ""

                    if want_stim:
                        command_label = f"stim_target_{predicted_target}_{stim_duration_ms}ms"

                        if not stimulator_enabled or stimulator is None:
                            stim_error = "stimulator_not_enabled"
                            timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                            print(f'{timestamp_str} ❌ stimulator start 跳过 | decode_id:{decode_id} | '
                                  f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                  f'原因:stimulator未启用或未连接')

                        elif stim_window_active:
                            stim_error = "stim_is_busy"
                            timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                            print(f'{timestamp_str} ❌ stimulator start 跳过 | decode_id:{decode_id} | '
                                  f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                  f'原因:刺激窗口激活中 (stim_window_active)')

                        elif stim_thread.is_busy():
                            stim_error = "stim_is_busy"
                            timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                            print(f'{timestamp_str} ❌ stimulator start 跳过 | decode_id:{decode_id} | '
                                  f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                  f'原因:stim_thread忙碌中 (stim_thread.is_busy)')

                        else:
                            params = stim_param_map.get(predicted_target, None)

                            if params is None:
                                stim_error = "stim_params_missing"
                                timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                                print(f'{timestamp_str} ❌ stimulator start 跳过 | decode_id:{decode_id} | '
                                      f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                      f'原因:参数缺失 (stim_params_missing)')
                            else:
                                stim_submit_ok = int(
                                    stim_thread.submit(
                                        decode_id=decode_id,
                                        command_type="start",
                                        params=params,
                                        command_label=command_label,
                                    )
                                )

                                if stim_submit_ok:
                                    stim_window_active = True
                                    stim_window_decode_id = decode_id
                                    stim_window_row_idx = None
                                    stim_window_command_label = command_label
                                    stim_window_params = params
                                    stim_window_deadline_s = action_now_s + stim_duration_ms / 1000.0
                                    stim_stop_pending = False
                                    stim_stop_decode_id = None

                                    # 打印提交成功信息和时间窗口
                                    timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                                    print(f'{timestamp_str} 📤 stimulator start 已提交 | decode_id:{decode_id} | '
                                          f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                          f'命令:{command_label} | 刺激窗口:{action_now_s:.3f}s - {stim_window_deadline_s:.3f}s '
                                          f'(持续{stim_duration_ms}ms)')

                                else:
                                    stim_error = "stim_is_busy"
                                    # 打印提交失败信息
                                    timestamp_str = datetime.now().strftime("[%H:%M:%S]")
                                    print(f'{timestamp_str} ❌ stimulator start 提交失败 | decode_id:{decode_id} | '
                                          f'预测:{predicted_target} | 置信度:{decode_result.get("confidence", 0):.4f} | '
                                          f'原因:stim_thread队列满或异常')

                    log_row = {
                        "decode_id": decode_id,
                        "participant": expInfo.get("participant", "unknown"),
                        "session": expInfo.get("session", "001"),
                        "action_start_time_s": round(action_start_time_s, 6),
                        "action_end_time_s": "",
                        "time_in_action_s": round(action_now_s - action_start_time_s, 6),
                        "data_received_time_s": decode_payload.get("data_received_time_s", ""),
                        "new_chunk_shape": new_chunk_shape,
                        "data_shape": decode_payload.get("data_shape", ""),
                        "preprocess_time_ms": preprocess_time_ms,
                        "decode_time_ms": decode_time_ms,
                        "total_pipeline_time_ms": total_pipeline_time_ms,
                        "decode_result": predicted_target,
                        "confidence": decode_result.get("confidence", ""),
                        "should_stim": should_stim,
                        "stim_submit_ok": stim_submit_ok,
                        "command_sent": command_sent,
                        "command_content": command_label,
                        "stim_error": stim_error,
                    }

                    action_log_cache.append(log_row)

                    if stim_submit_ok:
                        action_pending_log_by_id[decode_id] = len(action_log_cache) - 1
                        stim_window_row_idx = len(action_log_cache) - 1

                # 4. 回填刺激线程结果
                stim_payload = stim_thread.consume_result()

                if stim_payload is not None:
                    decode_id = stim_payload.get("decode_id", None)
                    command_type = stim_payload.get("command_type", "")

                    if decode_id in action_pending_log_by_id:
                        row_idx = action_pending_log_by_id[decode_id]

                        if 0 <= row_idx < len(action_log_cache):
                            if command_type == "start":
                                action_log_cache[row_idx]["command_sent"] = stim_payload.get("command_sent", 0)

                            returned_content = stim_payload.get("command_content", "")
                            if returned_content:
                                action_log_cache[row_idx]["command_content"] = returned_content

                            stim_error_text = stim_payload.get("error", "")
                            if stim_error_text:
                                old_error = action_log_cache[row_idx].get("stim_error", "")
                                if old_error:
                                    action_log_cache[row_idx]["stim_error"] = f"{old_error};{stim_error_text}"
                                else:
                                    action_log_cache[row_idx]["stim_error"] = stim_error_text

                    if command_type == "stop" and decode_id == stim_stop_decode_id:
                        stim_stop_pending = False
                        stim_stop_decode_id = None

                        if not stim_window_active:
                            stim_window_decode_id = None
                            stim_window_row_idx = None
                            stim_window_command_label = ""
                            stim_window_params = None
                            stim_window_deadline_s = 0.0

                # *act_movie* updates

                # if act_movie is starting this frame...
                if act_movie.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    act_movie.frameNStart = frameN  # exact frame index
                    act_movie.tStart = t  # local t and not account for scr refresh
                    act_movie.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(act_movie, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'act_movie.started')
                    # update status
                    act_movie.status = STARTED
                    act_movie.setAutoDraw(True)
                    act_movie.play()

                # if act_movie is stopping this frame...
                if act_movie.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > act_movie.tStartRefresh + 10 - frameTolerance or act_movie.isFinished:
                        # keep track of stop time/frame for later
                        act_movie.tStop = t  # not accounting for scr refresh
                        act_movie.tStopRefresh = tThisFlipGlobal  # on global time
                        act_movie.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'act_movie.stopped')
                        # update status
                        act_movie.status = FINISHED
                        act_movie.setAutoDraw(False)
                        act_movie.stop()
                if act_movie.isFinished:  # force-end the Routine
                    continueRoutine = False

                # *act_sound* updates

                # if act_sound is starting this frame...
                if act_sound.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    act_sound.frameNStart = frameN  # exact frame index
                    act_sound.tStart = t  # local t and not account for scr refresh
                    act_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('act_sound.started', tThisFlipGlobal)
                    # update status
                    act_sound.status = STARTED
                    act_sound.play(when=win)  # sync with win flip

                # if act_sound is stopping this frame...
                if act_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > act_sound.tStartRefresh + 3 - frameTolerance or act_sound.isFinished:
                        # keep track of stop time/frame for later
                        act_sound.tStop = t  # not accounting for scr refresh
                        act_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        act_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'act_sound.stopped')
                        # update status
                        act_sound.status = FINISHED
                        act_sound.stop()

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[act_movie, act_sound]
                    )
                    # skip the frame we paused on
                    continue

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    action.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in action.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "action" ---
            for thisComponent in action.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for action
            action.tStop = globalClock.getTime(format='float')
            action.tStopRefresh = tThisFlipGlobal
            thisExp.addData('action.stopped', action.tStop)
            # Run 'End Routine' code from act_code
            action_end_time_s = monotonic_time_s()

            # action 结束时，如果业务窗口还在，先直接关闭业务窗口
            if stim_window_active:
                stim_window_active = False

            # 如果 stop 还没提交，且有有效参数，则补提一次 stop
            if (
                    (not stim_stop_pending)
                    and stim_window_params is not None
                    and (not stim_thread.is_busy())
            ):
                stop_submit_ok = stim_thread.submit(
                    decode_id=stim_window_decode_id if stim_window_decode_id is not None else -1,
                    command_type="stop",
                    params=stim_window_params,
                    command_label=stim_window_command_label,
                )
                if stop_submit_ok:
                    stim_stop_pending = True
                    stim_stop_decode_id = stim_window_decode_id

            # 给线程一点时间回填 start/stop 结果
            drain_deadline = time.perf_counter() + 0.8

            while time.perf_counter() < drain_deadline:
                stim_payload = stim_thread.consume_result()

                if stim_payload is None:
                    core.wait(0.01)
                    continue

                decode_id = stim_payload.get("decode_id", None)
                command_type = stim_payload.get("command_type", "")

                if decode_id in action_pending_log_by_id:
                    row_idx = action_pending_log_by_id[decode_id]

                    if 0 <= row_idx < len(action_log_cache):
                        if command_type == "start":
                            action_log_cache[row_idx]["command_sent"] = stim_payload.get("command_sent", 0)

                        returned_content = stim_payload.get("command_content", "")
                        if returned_content:
                            action_log_cache[row_idx]["command_content"] = returned_content

                        stim_error_text = stim_payload.get("error", "")
                        if stim_error_text:
                            old_error = action_log_cache[row_idx].get("stim_error", "")
                            if old_error:
                                action_log_cache[row_idx]["stim_error"] = f"{old_error};{stim_error_text}"
                            else:
                                action_log_cache[row_idx]["stim_error"] = stim_error_text

                if command_type == "stop" and decode_id == stim_stop_decode_id:
                    stim_stop_pending = False
                    stim_stop_decode_id = None
                    stim_window_decode_id = None
                    stim_window_row_idx = None
                    stim_window_command_label = ""
                    stim_window_params = None
                    stim_window_deadline_s = 0.0

            # 如果 action 结束时 stop 还没回，不再把它当作业务 busy，只做标记
            if stim_stop_pending and stim_window_row_idx is not None:
                if 0 <= stim_window_row_idx < len(action_log_cache):
                    old_error = action_log_cache[stim_window_row_idx].get("stim_error", "")
                    if not old_error:
                        action_log_cache[stim_window_row_idx]["stim_error"] = "stop_result_pending_when_action_end"

            for row in action_log_cache:
                row["action_end_time_s"] = round(action_end_time_s, 6)

            try:
                save_action_decode_logs(action_log_cache, action_log_file_path)
            except Exception as e:
                print(f"save_action_decode_logs failed: {e}")

            act_movie.stop()  # ensure movie has stopped at end of Routine
            act_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if action.maxDurationReached:
                routineTimer.addTime(-action.maxDuration)
            elif action.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)

            # --- Prepare to start Routine "trial_over" ---
            # create an object to store info about Routine trial_over
            trial_over = data.Routine(
                name='trial_over',
                components=[to_text, to_sound],
            )
            trial_over.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from to_code
            to_tex = "复位"
            to_sound_dir = os.path.join(sound_dir, "复位.wav")
            to_text.setText(to_tex)
            to_sound.setSound(to_sound_dir, secs=3, hamming=True)
            to_sound.setVolume(1.0, log=False)
            to_sound.seek(0)
            # store start times for trial_over
            trial_over.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_over.tStart = globalClock.getTime(format='float')
            trial_over.status = STARTED
            thisExp.addData('trial_over.started', trial_over.tStart)
            trial_over.maxDuration = None
            # keep track of which components have finished
            trial_overComponents = trial_over.components
            for thisComponent in trial_over.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "trial_over" ---
            # if trial has changed, end Routine now
            if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
                continueRoutine = False
            trial_over.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *to_text* updates

                # if to_text is starting this frame...
                if to_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    to_text.frameNStart = frameN  # exact frame index
                    to_text.tStart = t  # local t and not account for scr refresh
                    to_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(to_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'to_text.started')
                    # update status
                    to_text.status = STARTED
                    to_text.setAutoDraw(True)

                # if to_text is active this frame...
                if to_text.status == STARTED:
                    # update params
                    pass

                # if to_text is stopping this frame...
                if to_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > to_text.tStartRefresh + 5 - frameTolerance:
                        # keep track of stop time/frame for later
                        to_text.tStop = t  # not accounting for scr refresh
                        to_text.tStopRefresh = tThisFlipGlobal  # on global time
                        to_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'to_text.stopped')
                        # update status
                        to_text.status = FINISHED
                        to_text.setAutoDraw(False)

                # *to_sound* updates

                # if to_sound is starting this frame...
                if to_sound.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    to_sound.frameNStart = frameN  # exact frame index
                    to_sound.tStart = t  # local t and not account for scr refresh
                    to_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('to_sound.started', tThisFlipGlobal)
                    # update status
                    to_sound.status = STARTED
                    to_sound.play(when=win)  # sync with win flip

                # if to_sound is stopping this frame...
                if to_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > to_sound.tStartRefresh + 3 - frameTolerance or to_sound.isFinished:
                        # keep track of stop time/frame for later
                        to_sound.tStop = t  # not accounting for scr refresh
                        to_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        to_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'to_sound.stopped')
                        # update status
                        to_sound.status = FINISHED
                        to_sound.stop()

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[to_sound]
                    )
                    # skip the frame we paused on
                    continue

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial_over.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_over.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "trial_over" ---
            for thisComponent in trial_over.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_over
            trial_over.tStop = globalClock.getTime(format='float')
            trial_over.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_over.stopped', trial_over.tStop)
            to_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_over.maxDurationReached:
                routineTimer.addTime(-trial_over.maxDuration)
            elif trial_over.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)

            # --- Prepare to start Routine "inter_trial_interval" ---
            # create an object to store info about Routine inter_trial_interval
            inter_trial_interval = data.Routine(
                name='inter_trial_interval',
                components=[iti_text],
            )
            inter_trial_interval.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from iti_code
            import random
            # random.seed(random_seed)
            iti_duration = random.randint(2, 10)
            print(f"iti_duration: {iti_duration}")
            # store start times for inter_trial_interval
            inter_trial_interval.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            inter_trial_interval.tStart = globalClock.getTime(format='float')
            inter_trial_interval.status = STARTED
            thisExp.addData('inter_trial_interval.started', inter_trial_interval.tStart)
            inter_trial_interval.maxDuration = None
            # keep track of which components have finished
            inter_trial_intervalComponents = inter_trial_interval.components
            for thisComponent in inter_trial_interval.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1

            # --- Run Routine "inter_trial_interval" ---
            # if trial has changed, end Routine now
            if isinstance(trial_loop, data.TrialHandler2) and thisTrial_loop.thisN != trial_loop.thisTrial.thisN:
                continueRoutine = False
            inter_trial_interval.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *iti_text* updates

                # if iti_text is starting this frame...
                if iti_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    iti_text.frameNStart = frameN  # exact frame index
                    iti_text.tStart = t  # local t and not account for scr refresh
                    iti_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(iti_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_text.started')
                    # update status
                    iti_text.status = STARTED
                    iti_text.setAutoDraw(True)

                # if iti_text is active this frame...
                if iti_text.status == STARTED:
                    # update params
                    pass

                # if iti_text is stopping this frame...
                if iti_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > iti_text.tStartRefresh + iti_duration - frameTolerance:
                        # keep track of stop time/frame for later
                        iti_text.tStop = t  # not accounting for scr refresh
                        iti_text.tStopRefresh = tThisFlipGlobal  # on global time
                        iti_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'iti_text.stopped')
                        # update status
                        iti_text.status = FINISHED
                        iti_text.setAutoDraw(False)

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    inter_trial_interval.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in inter_trial_interval.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "inter_trial_interval" ---
            for thisComponent in inter_trial_interval.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for inter_trial_interval
            inter_trial_interval.tStop = globalClock.getTime(format='float')
            inter_trial_interval.tStopRefresh = tThisFlipGlobal
            thisExp.addData('inter_trial_interval.stopped', inter_trial_interval.tStop)
            # the Routine "inter_trial_interval" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()

        # completed trial_repeats repeats of 'trial_loop'

        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()

        # --- Prepare to start Routine "block_over" ---
        # create an object to store info about Routine block_over
        block_over = data.Routine(
            name='block_over',
            components=[bo_text, bo_sound],
        )
        block_over.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from bo_code
        bo_tex = "回合结束\n请休息"
        bo_sound_dir = os.path.join(sound_dir, "回合结束_请休息.wav")

        bo_text.setText(bo_tex)
        bo_sound.setSound(bo_sound_dir, secs=3, hamming=True)
        bo_sound.setVolume(1.0, log=False)
        bo_sound.seek(0)
        # store start times for block_over
        block_over.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_over.tStart = globalClock.getTime(format='float')
        block_over.status = STARTED
        thisExp.addData('block_over.started', block_over.tStart)
        block_over.maxDuration = None
        # keep track of which components have finished
        block_overComponents = block_over.components
        for thisComponent in block_over.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1

        # --- Run Routine "block_over" ---
        # if trial has changed, end Routine now
        if isinstance(block_loop, data.TrialHandler2) and thisBlock_loop.thisN != block_loop.thisTrial.thisN:
            continueRoutine = False
        block_over.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            # *bo_text* updates

            # if bo_text is starting this frame...
            if bo_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                # keep track of start time/frame for later
                bo_text.frameNStart = frameN  # exact frame index
                bo_text.tStart = t  # local t and not account for scr refresh
                bo_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bo_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bo_text.started')
                # update status
                bo_text.status = STARTED
                bo_text.setAutoDraw(True)

            # if bo_text is active this frame...
            if bo_text.status == STARTED:
                # update params
                pass

            # if bo_text is stopping this frame...
            if bo_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bo_text.tStartRefresh + 5 - frameTolerance:
                    # keep track of stop time/frame for later
                    bo_text.tStop = t  # not accounting for scr refresh
                    bo_text.tStopRefresh = tThisFlipGlobal  # on global time
                    bo_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bo_text.stopped')
                    # update status
                    bo_text.status = FINISHED
                    bo_text.setAutoDraw(False)

            # *bo_sound* updates

            # if bo_sound is starting this frame...
            if bo_sound.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                # keep track of start time/frame for later
                bo_sound.frameNStart = frameN  # exact frame index
                bo_sound.tStart = t  # local t and not account for scr refresh
                bo_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('bo_sound.started', tThisFlipGlobal)
                # update status
                bo_sound.status = STARTED
                bo_sound.play(when=win)  # sync with win flip

            # if bo_sound is stopping this frame...
            if bo_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bo_sound.tStartRefresh + 3 - frameTolerance or bo_sound.isFinished:
                    # keep track of stop time/frame for later
                    bo_sound.tStop = t  # not accounting for scr refresh
                    bo_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    bo_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bo_sound.stopped')
                    # update status
                    bo_sound.status = FINISHED
                    bo_sound.stop()

            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[bo_sound]
                )
                # skip the frame we paused on
                continue

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_over.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_over.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # --- Ending Routine "block_over" ---
        for thisComponent in block_over.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_over
        block_over.tStop = globalClock.getTime(format='float')
        block_over.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_over.stopped', block_over.tStop)
        bo_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if block_over.maxDurationReached:
            routineTimer.addTime(-block_over.maxDuration)
        elif block_over.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        thisExp.nextEntry()

    # completed block_repeats repeats of 'block_loop'

    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()

    # --- Prepare to start Routine "finish" ---
    # create an object to store info about Routine finish
    finish = data.Routine(
        name='finish',
        components=[finish_text, key_resp],
    )
    finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for finish
    finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    finish.tStart = globalClock.getTime(format='float')
    finish.status = STARTED
    thisExp.addData('finish.started', finish.tStart)
    finish.maxDuration = None
    # keep track of which components have finished
    finishComponents = finish.components
    for thisComponent in finish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "finish" ---
    finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *finish_text* updates

        # if finish_text is starting this frame...
        if finish_text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            finish_text.frameNStart = frameN  # exact frame index
            finish_text.tStart = t  # local t and not account for scr refresh
            finish_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(finish_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'finish_text.started')
            # update status
            finish_text.status = STARTED
            finish_text.setAutoDraw(True)

        # if finish_text is active this frame...
        if finish_text.status == STARTED:
            # update params
            pass

        # if finish_text is stopping this frame...
        if finish_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > finish_text.tStartRefresh + 5 - frameTolerance:
                # keep track of stop time/frame for later
                finish_text.tStop = t  # not accounting for scr refresh
                finish_text.tStopRefresh = tThisFlipGlobal  # on global time
                finish_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'finish_text.stopped')
                # update status
                finish_text.status = FINISHED
                finish_text.setAutoDraw(False)

        # *key_resp* updates
        waitOnFlip = False

        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip

        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 5 - frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.stopped')
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False

        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp,
                win=win,
                timers=[routineTimer],
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            finish.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "finish" ---
    for thisComponent in finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for finish
    finish.tStop = globalClock.getTime(format='float')
    finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('finish.stopped', finish.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys', key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if finish.maxDurationReached:
        routineTimer.addTime(-finish.maxDuration)
    elif finish.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from init_code
    cleanup_all_resources()

    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment

    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.

    This function does NOT close the window or end the Python process - use `quit` for this.

    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip()
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.

    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo,
        thisExp=thisExp,
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
