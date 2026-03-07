# -*- coding: utf-8 -*-
"""
本模块用于处理 PsychoPy 的退出逻辑。
通过 "猴子补丁" (Monkey-Patching) 的方式，
它允许我们在原始的 core.quit() 函数执行前，先运行自定义的清理函数。
"""

from psychopy import core, logging

# 用于保存原始的 core.quit 函数
_original_quit = None

# 用于注册我们希望在退出时执行的清理函数
_cleanup_functions = []


def register_cleanup_function(func, *args, **kwargs):
    """
    注册一个清理函数。
    :param func: 需要在退出时调用的函数。
    :param args: 传递给该函数的位置参数。
    :param kwargs: 传递给该函数的关键字参数。
    """
    _cleanup_functions.append({'func': func, 'args': args, 'kwargs': kwargs})
    logging.info(f"Function '{func.__name__}' registered for cleanup.")


def _custom_quit():
    """这是我们自定义的退出函数，它将替换掉原始的 core.quit。"""
    logging.critical("--- Custom quit handler triggered (e.g., by ESC key) ---")
    # 逆序执行清理函数，这是一种常见的清理模式
    for item in reversed(_cleanup_functions):
        try:
            item['func'](*item['args'], **item['kwargs'])
        except Exception as e:
            logging.error(f"Error during cleanup function '{item['func'].__name__}': {e}")
    
    # 执行完我们的清理逻辑后，再调用原始的退出函数
    if _original_quit:
        _original_quit()


def patch_core_quit():
    """执行替换操作，将 core.quit 指向我们的自定义函数。"""
    global _original_quit
    if core.quit.__name__ != '_custom_quit':
        _original_quit = core.quit
        core.quit = _custom_quit
        logging.info("core.quit has been patched to include custom cleanup.")