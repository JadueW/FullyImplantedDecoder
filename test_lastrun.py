#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.5),
    on 二月 27, 2026, at 13:46
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
# 全局初始化

import sys
import os
import json
import warnings
from psychopy import logging, core

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 1. 导入项目模块
from transmission.trans_manager.remoteManagerTTL import RemoteManagerTTL
from peripheral.camera.MultiCamThreadRecord import (
    init_camera_thread,
    start_camera_thread,
    exit_camera_thread
)
from peripheral.audio.AudioThreadRecord import (
    init_audio_thread,
    start_audio_thread,
    exit_audio_thread
)
from utils.exit_handler import patch_core_quit, register_cleanup_function

# 2. 读取配置文件
config_path = os.path.join(project_dir, 'config', 'upper_limb_movement_config.json')

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info(f"✓ Config loaded from {config_path}")
except Exception as e:
    logging.error(f"✗ Failed to load config: {e}")
    config = {}
    # 使用默认值
    warnings.warn("Using default configuration values")
    
# 3. 获取配置参数
## 3.1 TCP 连接
host = config.get("experiment", {}).get("host", "127.0.0.1")
com_port = config.get("experiment", {}).get("com_port", "COM7")

## 3.2 文本大小
big_text_size = config.get("experiment", {}).get("big_text_size", 0.25)
small_text_size = config.get("experiment", {}).get("small_text_size", 0.25)

## 3.3 循环次数
trial_repeats = config.get("experiment",{}).get("trial_repetas",1)
block_repeats = config.get("experiment",{}).get("block_repeats",1)

## 3.4 任务类型
task_type = config.get("experiment", {}).get("task_type", 1)
task_desc = config.get("experiment", {}).get("task_desc", {"0": "粗大", "1": "精细"})

if task_type==1:
    selected_rows = "4:7"
    task_num =  3
else:
    selected_rows = "0:4"
    task_num = 4
    if task_type != 0:
        print("WARNING: 任务类型只能为0或1，默认为0.")

## 3.5条件文件路径
conditions_file = config.get("experiment", {}).get(
    "conditions",
    "D:/dev/paradigm_dev/resources/conditions/upper_limb_movement/condition.csv"
)

block_conditions_file = config.get("experiment", {}).get(
    "block_conditions",
    "D:/dev/paradigm_dev/resources/conditions/upper_limb_movement/block_conditions.csv"
)

## 3.6 随机种子
random_seed = config.get("experiment", {}).get("random_seed", 42)

## 3.7 日志输出
logging.info(f"Experiment configuration:")
logging.info(f"  Host: {host}")
logging.info(f"  COM port: {com_port}")
logging.info(f"  Task type: {task_type} ({task_desc.get(str(task_type), 'unknown')})")
logging.info(f"  Block repeats: {block_repeats}")
logging.info(f"  Trial repeats: {trial_repeats}")

## 3.8 判断外设的使能情况
### camera
camera_connected_status = config.get("experiment",{}).get("peripheral_stats",{}).get("is_camera_connected",False)

### audio
audio_connected_status = config.get("experiment",{}).get("peripheral_stats",{}).get("is_audio_connected",False)


# 4. 初始化 RemoteManagerTTL
rm = None
marker_mode = 'bit'  # 使用位模式标记
try:
    rm = RemoteManagerTTL(
        host=host,
        com_port=com_port,
        baudrate=115200
    )
    rm.initialize_device(mode=0)
    rm.begin_collect()
    logging.info("✓ RemoteManagerTTL initialized and started")
except Exception as e:
    logging.error(f"✗ RemoteManagerTTL initialization failed: {e}")
    warnings.warn(f"Could not initialize RemoteManager: {e}")
    print(f"ERROR: {e}")
    print("Experiment will continue without marker functionality")

# 5. 初始化相机录制
camera_threads = None
if camera_connected_status:
    try:
        from peripheral.camera.MultiCamThreadRecord import Config as CameraConfig
        camera_config_file = os.path.join(project_dir, 'config', 'upper_limb_movement_config.json')
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

# 7. 注册清理函数
def cleanup_all_resources():
    logging.info("=" * 50)
    logging.info("Cleaning up resources...")

    # 1. 停止 RemoteManager
    if rm:
        try:
            rm.close_connection(stop_collect=True)
            logging.info("✓ RemoteManager closed")
        except Exception as e:
            logging.error(f"✗ Error closing RemoteManager: {e}")

    # 2. 停止音频录制
    if audio_thread:
        try:
            exit_audio_thread(audio_thread)
            logging.info("✓ Audio recording stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping audio: {e}")

    # 3. 停止相机录制
    if camera_threads:
        try:
            exit_camera_thread(camera_threads)
            logging.info("✓ Camera recording stopped")
        except Exception as e:
            logging.error(f"✗ Error stopping cameras: {e}")

    logging.info("=" * 50)
    logging.info("All resources cleaned up")

## 7.1 使用上面自定义的清理函数
register_cleanup_function(cleanup_all_resources)

# 8. 打补丁到 PsychoPy 的 core.quit()
patch_core_quit()
logging.info("✓ Cleanup handlers registered and patched")

# 9. 设置随机种子
import random
import numpy as np
random.seed(random_seed)
np.random.seed(random_seed)
logging.info(f"✓ Random seed set to {random_seed}")

# 10. 全局变量初始化用于后续的Routine
## 10.1 block相关
block_num = 0
block_start_time = 0.0
block_duration = 0.0

## 10.2 trial相关
trial_num = 0
stimulus_type = ""

## 10.3 trail_ready相关
prepare_start_time = 0.0
prepare_duration = 0.0

# 10.4 action 相关
action_start_time = 0.0
action_end_time = 0.0
action_duration = 0.0
action_timer = None
action_prompt = ""

# trial_over 相关
reset_start_time = 0.0
reset_duration = 0.0
reset_prompt = ""

# block_over 相关
rest_start_time = 0.0
rest_duration = 0.0

logging.info("=" * 50)
logging.info("Upper Limb Movement Paradigm Initialized")
logging.info(f"Task: {task_desc.get(str(task_type), 'unknown')}")
logging.info("=" * 50)

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.5'
expName = 'uper_limb_movement_exp'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': 'data.getDateStr()',
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
_winSize = [2560, 1440]
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
        originPath='D:\\dev\\paradigm_dev\\paradigm\\test_lastrun.py',
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
    logFile = logging.LogFile(filename+'.log')
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
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
    # create speaker 'block_sound_5'
    deviceManager.addDevice(
        deviceName='block_sound_5',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'sound_2'
    deviceManager.addDevice(
        deviceName='sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'action_sound'
    deviceManager.addDevice(
        deviceName='action_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'trial_over_sound'
    deviceManager.addDevice(
        deviceName='trial_over_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'block_end_sound'
    deviceManager.addDevice(
        deviceName='block_end_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('end_key_resp') is None:
        # initialise end_key_resp
        end_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_key_resp',
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
    init_text = visual.TextStim(win=win, name='init_text',
        text='upper_limb_movement\nInitialization',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "block_ready" ---
    block_ready_text = visual.TextStim(win=win, name='block_ready_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    block_sound_5 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='block_sound_5',    name='block_sound_5'
    )
    block_sound_5.setVolume(1.0)
    
    # --- Initialize components for Routine "trial_ready" ---
    trial_ready_text = visual.TextStim(win=win, name='trial_ready_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.2078, 0.6078, -0.6078], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_2 = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "action" ---
    action_text = visual.TextStim(win=win, name='action_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    action_sound = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='action_sound',    name='action_sound'
    )
    action_sound.setVolume(1.0)
    # Run 'Begin Experiment' code from tqdm
    from tqdm import tqdm
    global trial_bar 
    print("block_repeats", block_repeats)
    print("trial_repeats", trial_repeats)
    
    total_trials = block_repeats * trial_repeats * task_num
    
    
    # 初始化进度条，描述文本将在后面动态更新
    trial_bar = tqdm(total=total_trials, desc="实验准备中...")
    
    
    print("begin exp")
    
    
    # --- Initialize components for Routine "trial_over" ---
    trial_over_text = visual.TextStim(win=win, name='trial_over_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_over_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='trial_over_sound',    name='trial_over_sound'
    )
    trial_over_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "block_over" ---
    block_end_text = visual.TextStim(win=win, name='block_end_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    block_end_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='block_end_sound',    name='block_end_sound'
    )
    block_end_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "finished" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='结束 \n谢谢配合\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_key_resp = keyboard.Keyboard(deviceName='end_key_resp')
    
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
    # Run 'Begin Routine' code from init_code
    init_time = core.monotonicClock.getTime()
    thisExp.addData('init_time', init_time)
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
        if init_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
            if tThisFlipGlobal > init_text.tStartRefresh + 3-frameTolerance:
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
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        block_conditions_file, 
        selection='0'
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
            components=[block_ready_text, block_sound_5],
        )
        block_ready.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        block_ready_text.setText(block_stimuli)
        block_sound_5.setSound(block_dir, secs=5, hamming=True)
        block_sound_5.setVolume(1.0, log=False)
        block_sound_5.seek(0)
        # Run 'Begin Routine' code from code
        # 设置block开始标记，设置marker通道为4
        
        if rm:
            rm.set_marker(4, mode=marker_mode)
            print(f"✓ Block {thisBlock_loop.thisN + 1} started at {core.monotonicClock.getTime():.3f}s")
            print(f"  Block stimuli: {block_stimuli}")
        else:
            print("WARNING: rm is None, marker not set")
        
        # 记录Block开始时间
        block_start_time = core.monotonicClock.getTime()
        thisExp.addData('block_start_time', block_start_time)
        
        
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
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_ready_text* updates
            
            # if block_ready_text is starting this frame...
            if block_ready_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_ready_text.frameNStart = frameN  # exact frame index
                block_ready_text.tStart = t  # local t and not account for scr refresh
                block_ready_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_ready_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_ready_text.started')
                # update status
                block_ready_text.status = STARTED
                block_ready_text.setAutoDraw(True)
            
            # if block_ready_text is active this frame...
            if block_ready_text.status == STARTED:
                # update params
                pass
            
            # if block_ready_text is stopping this frame...
            if block_ready_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_ready_text.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    block_ready_text.tStop = t  # not accounting for scr refresh
                    block_ready_text.tStopRefresh = tThisFlipGlobal  # on global time
                    block_ready_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'block_ready_text.stopped')
                    # update status
                    block_ready_text.status = FINISHED
                    block_ready_text.setAutoDraw(False)
            
            # *block_sound_5* updates
            
            # if block_sound_5 is starting this frame...
            if block_sound_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_sound_5.frameNStart = frameN  # exact frame index
                block_sound_5.tStart = t  # local t and not account for scr refresh
                block_sound_5.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('block_sound_5.started', tThisFlipGlobal)
                # update status
                block_sound_5.status = STARTED
                block_sound_5.play(when=win)  # sync with win flip
            
            # if block_sound_5 is stopping this frame...
            if block_sound_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_sound_5.tStartRefresh + 5-frameTolerance or block_sound_5.isFinished:
                    # keep track of stop time/frame for later
                    block_sound_5.tStop = t  # not accounting for scr refresh
                    block_sound_5.tStopRefresh = tThisFlipGlobal  # on global time
                    block_sound_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'block_sound_5.stopped')
                    # update status
                    block_sound_5.status = FINISHED
                    block_sound_5.stop()
            
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
                    playbackComponents=[block_sound_5]
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
        block_sound_5.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from code
        # 清除block的标记
        
        if rm:
            rm.set_marker(0, mode=marker_mode)
            print(f"✓ Block marker cleared at {core.monotonicClock.getTime():.3f}s")
        
        # 记录Block持续时间
        block_end_time = core.monotonicClock.getTime()
        block_duration = block_end_time - block_start_time
        thisExp.addData('block_duration', block_duration)
        print(f"  Block duration: {block_duration:.3f}s")
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if block_ready.maxDurationReached:
            routineTimer.addTime(-block_ready.maxDuration)
        elif block_ready.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # set up handler to look after randomisation of conditions etc
        trial_loop = data.TrialHandler2(
            name='trial_loop',
            nReps=trial_repeats, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            conditions_file, 
            selection=selected_rows
        )
        , 
            seed=random_seed, 
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
                components=[trial_ready_text, sound_2],
            )
            trial_ready.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            trial_ready_text.setText(stimuli)
            sound_2.setSound(dir, secs=2, hamming=True)
            sound_2.setVolume(1.0, log=False)
            sound_2.seek(0)
            # Run 'Begin Routine' code from trial_ready_code
            # 设置 marker 通道 1，用作提示
            rm.set_marker(1, mode=marker_mode)
            
            # 记录准备开始时间
            prepare_start_time = core.monotonicClock.getTime()
            thisExp.addData('prepare_start_time', prepare_start_time)
            thisExp.addData('stimulus_type', stimuli)
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
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_ready_text* updates
                
                # if trial_ready_text is starting this frame...
                if trial_ready_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_ready_text.frameNStart = frameN  # exact frame index
                    trial_ready_text.tStart = t  # local t and not account for scr refresh
                    trial_ready_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_ready_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_ready_text.started')
                    # update status
                    trial_ready_text.status = STARTED
                    trial_ready_text.setAutoDraw(True)
                
                # if trial_ready_text is active this frame...
                if trial_ready_text.status == STARTED:
                    # update params
                    pass
                
                # if trial_ready_text is stopping this frame...
                if trial_ready_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_ready_text.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_ready_text.tStop = t  # not accounting for scr refresh
                        trial_ready_text.tStopRefresh = tThisFlipGlobal  # on global time
                        trial_ready_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_ready_text.stopped')
                        # update status
                        trial_ready_text.status = FINISHED
                        trial_ready_text.setAutoDraw(False)
                
                # *sound_2* updates
                
                # if sound_2 is starting this frame...
                if sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sound_2.frameNStart = frameN  # exact frame index
                    sound_2.tStart = t  # local t and not account for scr refresh
                    sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_2.started', tThisFlipGlobal)
                    # update status
                    sound_2.status = STARTED
                    sound_2.play(when=win)  # sync with win flip
                
                # if sound_2 is stopping this frame...
                if sound_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_2.tStartRefresh + 2-frameTolerance or sound_2.isFinished:
                        # keep track of stop time/frame for later
                        sound_2.tStop = t  # not accounting for scr refresh
                        sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                        sound_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_2.stopped')
                        # update status
                        sound_2.status = FINISHED
                        sound_2.stop()
                
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
                        playbackComponents=[sound_2]
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
            sound_2.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from trial_ready_code
            # 清除所有 marker
            rm.set_marker(0, mode=marker_mode)
            
            # 记录准备持续时间
            prepare_end_time = core.monotonicClock.getTime()
            prepare_duration = prepare_end_time - prepare_start_time
            thisExp.addData('prepare_duration', prepare_duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_ready.maxDurationReached:
                routineTimer.addTime(-trial_ready.maxDuration)
            elif trial_ready.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "action" ---
            # create an object to store info about Routine action
            action = data.Routine(
                name='action',
                components=[action_text, action_sound],
            )
            action.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            action_text.setText('开始')
            action_sound.setSound('D:/dev/paradigm_dev/resources/audios/upper_limb_movement/开始.wav', secs=1.0, hamming=True)
            action_sound.setVolume(1.0, log=False)
            action_sound.seek(0)
            # Run 'Begin Routine' code from tqdm
            #打标 在第1个通道打标
            rm.set_marker(2, mode=marker_mode)
            # 记录动作开始时间
            action_start_time = core.monotonicClock.getTime()
            thisExp.addData('action_start_time', action_start_time)
            
            # 1. 首先，为即将开始的这个 trial 更新进度条
            #    让进度条的数值（如 11/25 -> 12/25）先行一步
            trial_bar.update(1)
            
            # 2. 然后，检查这是否是一个新 block 的开始
            #    PsychoPy 会自动创建名为 block_loop 和 trials 的循环处理器
            if trial_loop.thisN == 0:
                # 获取当前 block 的编号 (thisN 从0开始，所以+1)
                current_block = block_loop.thisN + 1
                # 获取总 block 数
                total_blocks = block_loop.nTotal
                
                # 构建新的描述文本
                desc_text = f"block进度：{current_block}/{total_blocks}，总进度"
                
                # 更新描述。这会再次刷新屏幕，但此时进度条的数值已经是正确的了
                trial_bar.set_description(desc_text)
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
                
                # *action_text* updates
                
                # if action_text is starting this frame...
                if action_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    action_text.frameNStart = frameN  # exact frame index
                    action_text.tStart = t  # local t and not account for scr refresh
                    action_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(action_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'action_text.started')
                    # update status
                    action_text.status = STARTED
                    action_text.setAutoDraw(True)
                
                # if action_text is active this frame...
                if action_text.status == STARTED:
                    # update params
                    pass
                
                # if action_text is stopping this frame...
                if action_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > action_text.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        action_text.tStop = t  # not accounting for scr refresh
                        action_text.tStopRefresh = tThisFlipGlobal  # on global time
                        action_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'action_text.stopped')
                        # update status
                        action_text.status = FINISHED
                        action_text.setAutoDraw(False)
                
                # *action_sound* updates
                
                # if action_sound is starting this frame...
                if action_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    action_sound.frameNStart = frameN  # exact frame index
                    action_sound.tStart = t  # local t and not account for scr refresh
                    action_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('action_sound.started', tThisFlipGlobal)
                    # update status
                    action_sound.status = STARTED
                    action_sound.play(when=win)  # sync with win flip
                
                # if action_sound is stopping this frame...
                if action_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > action_sound.tStartRefresh + 1.0-frameTolerance or action_sound.isFinished:
                        # keep track of stop time/frame for later
                        action_sound.tStop = t  # not accounting for scr refresh
                        action_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        action_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'action_sound.stopped')
                        # update status
                        action_sound.status = FINISHED
                        action_sound.stop()
                
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
                        playbackComponents=[action_sound]
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
            action_sound.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from tqdm
            rm.set_marker(0, mode=marker_mode)
            
            action_end_time = core.monotonicClock.getTime()
            action_duration = action_end_time - action_start_time
            thisExp.addData('action_end_time', action_end_time)
            thisExp.addData('action_duration', action_duration)
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
                components=[trial_over_text, trial_over_sound],
            )
            trial_over.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            trial_over_text.setText('复位')
            trial_over_sound.setSound('D:/dev/paradigm_dev/resources/audios/upper_limb_movement/复位.wav', secs=5, hamming=True)
            trial_over_sound.setVolume(1.0, log=False)
            trial_over_sound.seek(0)
            # Run 'Begin Routine' code from trial_over_code
            # 设置 marker 通道 3（复位提示）
            rm.set_marker(3, mode=marker_mode)
            
            # 记录复位开始时间
            reset_start_time = core.monotonicClock.getTime()
            thisExp.addData('reset_start_time', reset_start_time)
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
                
                # *trial_over_text* updates
                
                # if trial_over_text is starting this frame...
                if trial_over_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_over_text.frameNStart = frameN  # exact frame index
                    trial_over_text.tStart = t  # local t and not account for scr refresh
                    trial_over_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_over_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_over_text.started')
                    # update status
                    trial_over_text.status = STARTED
                    trial_over_text.setAutoDraw(True)
                
                # if trial_over_text is active this frame...
                if trial_over_text.status == STARTED:
                    # update params
                    pass
                
                # if trial_over_text is stopping this frame...
                if trial_over_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_over_text.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        trial_over_text.tStop = t  # not accounting for scr refresh
                        trial_over_text.tStopRefresh = tThisFlipGlobal  # on global time
                        trial_over_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_over_text.stopped')
                        # update status
                        trial_over_text.status = FINISHED
                        trial_over_text.setAutoDraw(False)
                
                # *trial_over_sound* updates
                
                # if trial_over_sound is starting this frame...
                if trial_over_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_over_sound.frameNStart = frameN  # exact frame index
                    trial_over_sound.tStart = t  # local t and not account for scr refresh
                    trial_over_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('trial_over_sound.started', tThisFlipGlobal)
                    # update status
                    trial_over_sound.status = STARTED
                    trial_over_sound.play(when=win)  # sync with win flip
                
                # if trial_over_sound is stopping this frame...
                if trial_over_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > trial_over_sound.tStartRefresh + 5-frameTolerance or trial_over_sound.isFinished:
                        # keep track of stop time/frame for later
                        trial_over_sound.tStop = t  # not accounting for scr refresh
                        trial_over_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        trial_over_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trial_over_sound.stopped')
                        # update status
                        trial_over_sound.status = FINISHED
                        trial_over_sound.stop()
                
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
                        playbackComponents=[trial_over_sound]
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
            trial_over_sound.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from trial_over_code
            # 设置 marker 通道 0
            rm.set_marker(0, mode=marker_mode)
            
            # 记录复位结束时间
            reset_end_time = core.monotonicClock.getTime()
            thisExp.addData('reset_end_time', reset_end_time)
            reset_duration = reset_end_time - reset_start_time
            thisExp.addData('reset_duration', reset_duration)
            
            
            # 输出单个试次完成的信息
            print(f"    Prepare: {prepare_duration:.3f}s")
            print(f"    Action: {action_duration:.3f}s")
            print(f"    Reset: {reset_duration:.3f}s")
            
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_over.maxDurationReached:
                routineTimer.addTime(-trial_over.maxDuration)
            elif trial_over.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            thisExp.nextEntry()
            
        # completed trial_repeats repeats of 'trial_loop'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trial_loop.trialList in ([], [None], None):
            params = []
        else:
            params = trial_loop.trialList[0].keys()
        # save data for this loop
        trial_loop.saveAsExcel(filename + '.xlsx', sheetName='trial_loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        trial_loop.saveAsText(filename + 'trial_loop.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "block_over" ---
        # create an object to store info about Routine block_over
        block_over = data.Routine(
            name='block_over',
            components=[block_end_text, block_end_sound],
        )
        block_over.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        block_end_text.setText('回合结束\n休息5秒')
        block_end_sound.setSound('D:/dev/paradigm_dev/resources/audios/upper_limb_movement/回合结束_请休息.wav', secs=5, hamming=True)
        block_end_sound.setVolume(1.0, log=False)
        block_end_sound.seek(0)
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
            
            # *block_end_text* updates
            
            # if block_end_text is starting this frame...
            if block_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_end_text.frameNStart = frameN  # exact frame index
                block_end_text.tStart = t  # local t and not account for scr refresh
                block_end_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_end_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_end_text.started')
                # update status
                block_end_text.status = STARTED
                block_end_text.setAutoDraw(True)
            
            # if block_end_text is active this frame...
            if block_end_text.status == STARTED:
                # update params
                pass
            
            # if block_end_text is stopping this frame...
            if block_end_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_end_text.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    block_end_text.tStop = t  # not accounting for scr refresh
                    block_end_text.tStopRefresh = tThisFlipGlobal  # on global time
                    block_end_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'block_end_text.stopped')
                    # update status
                    block_end_text.status = FINISHED
                    block_end_text.setAutoDraw(False)
            
            # *block_end_sound* updates
            
            # if block_end_sound is starting this frame...
            if block_end_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_end_sound.frameNStart = frameN  # exact frame index
                block_end_sound.tStart = t  # local t and not account for scr refresh
                block_end_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('block_end_sound.started', tThisFlipGlobal)
                # update status
                block_end_sound.status = STARTED
                block_end_sound.play(when=win)  # sync with win flip
            
            # if block_end_sound is stopping this frame...
            if block_end_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_end_sound.tStartRefresh + 5-frameTolerance or block_end_sound.isFinished:
                    # keep track of stop time/frame for later
                    block_end_sound.tStop = t  # not accounting for scr refresh
                    block_end_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    block_end_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'block_end_sound.stopped')
                    # update status
                    block_end_sound.status = FINISHED
                    block_end_sound.stop()
            
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
                    playbackComponents=[block_end_sound]
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
        block_end_sound.pause()  # ensure sound has stopped at end of Routine
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
    # get names of stimulus parameters
    if block_loop.trialList in ([], [None], None):
        params = []
    else:
        params = block_loop.trialList[0].keys()
    # save data for this loop
    block_loop.saveAsExcel(filename + '.xlsx', sheetName='block_loop',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    block_loop.saveAsText(filename + 'block_loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "finished" ---
    # create an object to store info about Routine finished
    finished = data.Routine(
        name='finished',
        components=[end_text, end_key_resp],
    )
    finished.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_key_resp
    end_key_resp.keys = []
    end_key_resp.rt = []
    _end_key_resp_allKeys = []
    # store start times for finished
    finished.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    finished.tStart = globalClock.getTime(format='float')
    finished.status = STARTED
    thisExp.addData('finished.started', finished.tStart)
    finished.maxDuration = None
    # keep track of which components have finished
    finishedComponents = finished.components
    for thisComponent in finished.components:
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
    
    # --- Run Routine "finished" ---
    finished.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # if end_text is stopping this frame...
        if end_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end_text.tStop = t  # not accounting for scr refresh
                end_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_text.stopped')
                # update status
                end_text.status = FINISHED
                end_text.setAutoDraw(False)
        
        # *end_key_resp* updates
        waitOnFlip = False
        
        # if end_key_resp is starting this frame...
        if end_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_key_resp.frameNStart = frameN  # exact frame index
            end_key_resp.tStart = t  # local t and not account for scr refresh
            end_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_key_resp.started')
            # update status
            end_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if end_key_resp is stopping this frame...
        if end_key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_key_resp.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end_key_resp.tStop = t  # not accounting for scr refresh
                end_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                end_key_resp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_key_resp.stopped')
                # update status
                end_key_resp.status = FINISHED
                end_key_resp.status = FINISHED
        if end_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = end_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_key_resp_allKeys.extend(theseKeys)
            if len(_end_key_resp_allKeys):
                end_key_resp.keys = _end_key_resp_allKeys[-1].name  # just the last key pressed
                end_key_resp.rt = _end_key_resp_allKeys[-1].rt
                end_key_resp.duration = _end_key_resp_allKeys[-1].duration
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
            finished.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in finished.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "finished" ---
    for thisComponent in finished.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for finished
    finished.tStop = globalClock.getTime(format='float')
    finished.tStopRefresh = tThisFlipGlobal
    thisExp.addData('finished.stopped', finished.tStop)
    # check responses
    if end_key_resp.keys in ['', [], None]:  # No response was made
        end_key_resp.keys = None
    thisExp.addData('end_key_resp.keys',end_key_resp.keys)
    if end_key_resp.keys != None:  # we had a response
        thisExp.addData('end_key_resp.rt', end_key_resp.rt)
        thisExp.addData('end_key_resp.duration', end_key_resp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if finished.maxDurationReached:
        routineTimer.addTime(-finished.maxDuration)
    elif finished.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
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
    expInfo = showExpInfoDlg(expInfo=expInfo)
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
