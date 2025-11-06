import threading
import queue
import sys
import os

# Ensure Tcl/Tk paths are set when running from a bundled app
_BUNDLE = getattr(sys, '_MEIPASS', None)
if _BUNDLE:
    # Support both 9.0 and 8.6 when bundled
    tcl90 = os.path.join(_BUNDLE, 'lib', 'tcl9.0')
    tk90  = os.path.join(_BUNDLE, 'lib', 'tk9.0')
    tcl86 = os.path.join(_BUNDLE, 'lib', 'tcl8.6')
    tk86  = os.path.join(_BUNDLE, 'lib', 'tk8.6')
    if os.path.isdir(tcl90):
        os.environ.setdefault('TCL_LIBRARY', tcl90)
    elif os.path.isdir(tcl86):
        os.environ.setdefault('TCL_LIBRARY', tcl86)
    if os.path.isdir(tk90):
        os.environ.setdefault('TK_LIBRARY', tk90)
    elif os.path.isdir(tk86):
        os.environ.setdefault('TK_LIBRARY', tk86)

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from wordhunt import WordHuntSolver, move_iphone_mirror_to_corner


class MainThreadRunner:
    """Dispatch callables to Tk's main thread and wait for result.

    macOS UI frameworks (HIToolbox/AppKit/Quartz event taps) are strict about
    main-thread usage. This runner lets worker threads execute such calls on
    the Tk main thread safely.
    """
    def __init__(self, root: tk.Tk):
        self.root = root

    def run_sync(self, func, *args, **kwargs):
        done = threading.Event()
        result = {"val": None, "err": None}

        def _invoke():
            try:
                result["val"] = func(*args, **kwargs)
            except BaseException as e:
                result["err"] = e
            finally:
                done.set()

        # Schedule on Tk main loop
        self.root.after(0, _invoke)
        done.wait()
        if result["err"] is not None:
            raise result["err"]
        return result["val"]


class StdoutRedirector:
    def __init__(self, text_widget: ScrolledText):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self._orig_write = sys.stdout.write

    def write(self, s):
        self.queue.put(s)

    def flush(self):
        pass

    def pump(self):
        try:
            while True:
                s = self.queue.get_nowait()
                self.text_widget.insert(tk.END, s)
                self.text_widget.see(tk.END)
        except queue.Empty:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WordHunt Solver")
        self.geometry("700x520")
        self.resizable(True, True)

        self.solver = None
        self.worker = None
        self.main_runner = MainThreadRunner(self)

        # Controls
        frm = ttk.Frame(self)
        frm.pack(fill=tk.X, padx=10, pady=8)

        self.var_manual = tk.BooleanVar(value=False)
        self.var_dry = tk.BooleanVar(value=False)
        self.var_debug_grid = tk.BooleanVar(value=False)
        self.var_debug_detect = tk.BooleanVar(value=False)

        ttk.Button(frm, text="Permissions", command=self.open_permissions).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Move to Corner", command=self.move_to_corner).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Start", command=self.start_solve).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Dry Run", command=lambda: self.start_solve(dry=True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Calibrate", command=self.start_calibrate).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Reload Model", command=self.reload_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Stop", command=self.stop_solve).pack(side=tk.LEFT, padx=4)

        opt = ttk.Frame(self)
        opt.pack(fill=tk.X, padx=10)
        ttk.Checkbutton(opt, text="Manual Start", variable=self.var_manual).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opt, text="Grid Overlay", variable=self.var_debug_grid).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(opt, text="Detect Debug", variable=self.var_debug_detect).pack(side=tk.LEFT, padx=6)

        # Log output
        self.txt = ScrolledText(self, height=24)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        # Redirect stdout
        self.redirector = StdoutRedirector(self.txt)
        sys.stdout = self.redirector
        self.after(80, self._pump_stdout)

        self._print_welcome()

    def _pump_stdout(self):
        self.redirector.pump()
        self.after(80, self._pump_stdout)

    def _init_solver(self):
        if self.solver is None:
            # Avoid any OpenCV windows from worker threads
            # Provide UI dispatcher so solver can run macOS APIs on main thread
            self.solver = WordHuntSolver(no_windows=True, ui_call=self.main_runner.run_sync)
        # Apply flags
        self.solver.manual_start = self.var_manual.get()
        # Set debug via env for called code paths
        os.environ['WORDHUNT_DEBUG_GRID'] = '1' if self.var_debug_grid.get() else '0'
        os.environ['WORDHUNT_DEBUG_DETECT'] = '1' if self.var_debug_detect.get() else '0'

    def move_to_corner(self):
        print("\nMoving iPhone window to corner...")
        # Ensure AppleScript menu interaction runs on the main thread
        try:
            self.main_runner.run_sync(move_iphone_mirror_to_corner)
        except Exception as e:
            print(f"Move to Corner error: {e}")

    def _print_welcome(self):
        print("""
Welcome! Quick steps:
1) Allow permissions (Screen Recording + Accessibility) using the Permissions button.
2) Ensure iPhone Mirroring is open and Word Hunt is on the Start screen.
3) Click Move to Corner, then Start. Use Manual Start if you prefer to tap Start yourself.
4) Optional: Calibrate once for perfect alignment. Calibration is saved.
""".strip())

    def check_permissions(self):
        # Screen Recording probe using mss
        screen_ok = False
        try:
            import mss, numpy as np
            sct = mss.mss()
            mon = sct.monitors[0]
            img = sct.grab(mon)
            arr = np.array(img)
            # Heuristic: if it's all zeros or near-constant, assume blocked
            screen_ok = bool(arr.std() > 1.0)
        except Exception as e:
            print(f"Screen Recording check error: {e}")
            screen_ok = False

        # Accessibility trust (AX API)
        access_ok = False
        try:
            try:
                from Quartz import AXIsProcessTrusted
                access_ok = bool(AXIsProcessTrusted())
            except Exception:
                from ctypes import cdll, c_bool
                app_services = cdll.LoadLibrary('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
                app_services.AXIsProcessTrusted.restype = c_bool
                access_ok = bool(app_services.AXIsProcessTrusted())
        except Exception as e:
            print(f"Accessibility check error: {e}")
            access_ok = False

        return screen_ok, access_ok

    def open_permissions(self):
        screen_ok, access_ok = self.check_permissions()
        print(f"Permissions → Screen Recording: {'OK' if screen_ok else 'NEEDED'}, Accessibility: {'OK' if access_ok else 'NEEDED'}")

        if not screen_ok:
            print("Opening Screen Recording settings…")
            os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenRecording"')
        if not access_ok:
            print("Opening Accessibility settings…")
            os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"')

        if not (screen_ok and access_ok):
            messagebox.showinfo(
                "Grant Permissions",
                "In System Settings > Privacy & Security, grant Screen Recording and Accessibility to this app (or Terminal during development), then restart the app."
            )

    def _run_solve(self, dry=False):
        try:
            self._init_solver()
            self.solver.no_swipe = dry
            self.solver.solve()
        except Exception as e:
            print(f"Error: {e}")

    def start_solve(self, dry=False):
        if self.worker and self.worker.is_alive():
            print("A run is already in progress.")
            return
        self.worker = threading.Thread(target=self._run_solve, args=(dry,), daemon=True)
        self.worker.start()

    def stop_solve(self):
        if self.solver is not None:
            self.solver.stop_flag = True
            print("Stop requested.")

    def _run_calibrate(self):
        try:
            self._init_solver()
            # Non-blocking countdown capture (3s per point), skip overlay to avoid cv2 window from worker thread
            self.solver.calibrate(delay_seconds=3.0, show_overlay=False)
            print("Calibration saved. You can enable Grid Overlay and run Dry Run to verify.")
        except Exception as e:
            print(f"Calibration error: {e}")

    def reload_model(self):
        def _do_reload():
            try:
                self._init_solver()
                # Reload respects WORDHUNT_MODEL or ~/.wordhunt override
                self.solver.reload_model()
            except Exception as e:
                print(f"Model reload error: {e}")
        # Run reload in a worker thread to keep UI snappy
        t = threading.Thread(target=_do_reload, daemon=True)
        t.start()

    def start_calibrate(self):
        if self.worker and self.worker.is_alive():
            print("Already running…")
            return
        self.worker = threading.Thread(target=self._run_calibrate, daemon=True)
        self.worker.start()


if __name__ == '__main__':
    App().mainloop()
