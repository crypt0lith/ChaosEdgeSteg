#!/usr/bin/env python
# coding:UTF-8
__all__ = ['header', 'Fore']

import ctypes
import os
import re


def enable_vt_processing() -> bool:
    _ansi_256_enabled = {'ANSICON', 'COLORTERM', 'ConEmuANSI', 'PYCHARM_HOSTED', 'TERM', 'TERMINAL_EMULATOR',
                         'TERM_PROGRAM', 'WT_SESSION'}
    _ansi_256_enabled.intersection_update(set(list(os.environ.keys())))
    if not any(_ansi_256_enabled):
        if os.name == 'nt':
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            if handle == -1:
                return False
            mode = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not ctypes.windll.kernel32.SetConsoleMode(handle, mode):
                return False
    return True


ANSI256_ENABLED = enable_vt_processing()


class Fore:
    RESET = '\033[0m'
    CYAN = '\033[38;5;14m' if ANSI256_ENABLED else '\033[36m'
    RED = '\033[38;5;9m' if ANSI256_ENABLED else '\033[31m'


def adjust_ansi_codes(__s: str):
    if not ANSI256_ENABLED:
        ansi_reg_codes = {
            '\033[38;5;9m': '\033[31m',
            '\033[38;5;15m': '\033[37m',
            '\033[38;5;8m': '\033[90m'
        }
        ansi_256_codes = re.findall(r'\x1B[@-_][0-?]*[ -/]*[@-~]', __s)
        ansi_256_unique = []
        for i in ansi_256_codes:
            if not ansi_reg_codes.get(i) or i in ansi_256_unique:
                continue
            ansi_256_unique.append(i)
            __s = __s.replace(i, ansi_reg_codes[i])
    return __s


def header():
    with open('header.bin', 'rb') as f:
        _header = str(f.read().decode('utf-8'))
    return adjust_ansi_codes(_header)


if __name__ == '__main__':
    print(header())
