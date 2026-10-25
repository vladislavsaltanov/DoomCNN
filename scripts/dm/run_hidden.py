"""Run a command on a separate, never-shown Windows desktop, so the game processes it starts (a new vizdoom.exe per
match) and their console windows cannot take the focus from the user's desktop. Waits for the command: start this
script itself detached.

    Start-Process .venv\\Scripts\\python -ArgumentList 'scripts\\dm\\run_hidden.py','runs\\dm\\e6_pipeline.cmd' -WindowStyle Hidden
"""
import ctypes
import ctypes.wintypes as w
import os
import sys

DESKTOP = 'doomcnn'
GENERIC_ALL = 0x10000000
INFINITE = 0xFFFFFFFF
CREATE_NEW_CONSOLE = 0x00000010


class STARTUPINFO(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR), ('lpTitle', w.LPWSTR),
                ('dwX', w.DWORD), ('dwY', w.DWORD), ('dwXSize', w.DWORD), ('dwYSize', w.DWORD),
                ('dwXCountChars', w.DWORD), ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD),
                ('dwFlags', w.DWORD), ('wShowWindow', w.WORD), ('cbReserved2', w.WORD), ('lpReserved2', w.LPBYTE),
                ('hStdInput', w.HANDLE), ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE), ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]


def run_on_desktop(cmdline, cwd):
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    user32.CreateDesktopW.restype = w.HANDLE
    desk = user32.CreateDesktopW(DESKTOP, None, None, 0, GENERIC_ALL, None)  # opens it if it exists
    if not desk:
        raise OSError(ctypes.get_last_error(), 'CreateDesktopW')
    si = STARTUPINFO(cb=ctypes.sizeof(STARTUPINFO), lpDesktop='WinSta0\\' + DESKTOP)
    pi = PROCESS_INFORMATION()
    buf = ctypes.create_unicode_buffer(cmdline)
    if not k32.CreateProcessW(None, buf, None, None, False, CREATE_NEW_CONSOLE, None, cwd, ctypes.byref(si),
                              ctypes.byref(pi)):
        raise OSError(ctypes.get_last_error(), 'CreateProcessW')
    k32.WaitForSingleObject(pi.hProcess, INFINITE)  # the desktop lives while this handle is open
    code = w.DWORD()
    k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    for h in (pi.hProcess, pi.hThread):
        k32.CloseHandle(h)
    user32.CloseDesktop(desk)
    return code.value


if __name__ == '__main__':
    target = sys.argv[1]
    cmd = f'cmd.exe /c "{target}"' if target.lower().endswith(('.cmd', '.bat')) else ' '.join(sys.argv[1:])
    sys.exit(run_on_desktop(cmd, os.getcwd()))
