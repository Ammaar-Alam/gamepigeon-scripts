import sys
import os
import threading
import queue
from time import sleep

from Cocoa import (
    NSApplication, NSApp, NSWindow, NSButton, NSTextView, NSScrollView,
    NSMakeRect, NSApplicationActivationPolicyRegular,
    NSRunningApplication, NSApplicationActivateIgnoringOtherApps,
    NSBezelStyleRounded, NSFont, NSTextField, NSTextFieldRoundedBezel,
    NSThread, NSProgressIndicator, NSView
)
try:
    # AppKit-only conveniences
    from AppKit import (
        NSVisualEffectView, NSAppearance, NSAppearanceNameDarkAqua, NSColor, NSImage,
        NSImageScaleProportionallyDown, NSImageLeft, NSImageSymbolConfiguration,
        NSVisualEffectMaterialSidebar, NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectStateActive, NSEventModifierFlagCommand, NSProgressIndicatorStyleSpinning,
        NSSegmentedControl
    )
except Exception:  # Fallback on older systems
    NSVisualEffectView = None
    NSAppearance = None
    NSAppearanceNameDarkAqua = None
    NSColor = None
    NSImage = None
    NSImageScaleProportionallyDown = 0
    NSImageLeft = 2
    NSImageSymbolConfiguration = None
    NSVisualEffectMaterialSidebar = 0
    NSVisualEffectBlendingModeBehindWindow = 0
    NSVisualEffectStateActive = 1
    NSEventModifierFlagCommand = 1 << 20
    NSProgressIndicatorStyleSpinning = 1
from PyObjCTools import AppHelper
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

from wordhunt import WordHuntSolver, move_iphone_mirror_to_corner


class Logger:
    def __init__(self, textview: NSTextView):
        self.textview = textview
        self.q = queue.Queue()
        self._orig = sys.stdout
        sys.stdout = self
        # Mirror logs to file when launched from Finder (no console)
        try:
            logs_dir = os.path.expanduser('~/Library/Logs')
            os.makedirs(logs_dir, exist_ok=True)
            self._log_path = os.path.join(logs_dir, 'WordHuntSolver.log')
            self._fh = open(self._log_path, 'a', buffering=1)
            print(f"Logging to {self._log_path}")
        except Exception:
            self._fh = None

    def write(self, s):
        self.q.put(s)
        if self._fh is not None:
            try:
                self._fh.write(s)
            except Exception:
                pass

    def flush(self):
        try:
            if self._fh is not None:
                self._fh.flush()
        except Exception:
            pass

    def pump(self):
        # Drain queue and append on main thread to avoid AppKit threading issues
        payload = []
        while True:
            try:
                payload.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not payload:
            return
        text = "".join(payload)

        def _append():
            self.textview.setString_(self.textview.string() + text)
            self.textview.scrollToEndOfDocument_(None)

        try:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(_append)
        except Exception:
            # Fallback (best effort)
            _append()


class App:
    def __init__(self):
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200.0, 300.0, 860.0, 520.0), 15, 2, False
        )
        self.window.setTitle_("WordHunt Solver")

        content = self.window.contentView()

        # Appearance & glass background
        try:
            if NSAppearance is not None and NSAppearanceNameDarkAqua is not None:
                self.window.setAppearance_(NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua))
        except Exception:
            pass

        try:
            if NSVisualEffectView is not None and os.getenv('WORDHUNT_NO_VIBRANCY') != '1':
                self.bg = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
                try:
                    self.bg.setMaterial_(NSVisualEffectMaterialSidebar)
                    self.bg.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
                    self.bg.setState_(NSVisualEffectStateActive)
                except Exception:
                    pass
                self.bg.setAutoresizingMask_(18)  # width + height sizable
                # Safer: simple addSubview to avoid ordering assertions
                content.addSubview_(self.bg)
        except Exception:
            pass

        # Buttons (toolbar-style)
        self.btnPerm = self._button(20, 460, 120, 30, "Permissions", "onPerm:")
        self.btnMove = self._button(150, 460, 130, 30, "Move to Corner", "onMove:")
        self.btnStart = self._button(290, 460, 90, 30, "Start", "onStart:")
        self.btnDry   = self._button(390, 460, 90, 30, "Dry Run", "onDry:")
        self.btnCal   = self._button(490, 460, 100, 30, "Calibrate", "onCal:")
        self.btnReload= self._button(600, 460, 120, 30, "Reload Model", "onReload:")
        self.btnStop  = self._button(725, 460, 80, 30, "Stop", "onStop:")

        # Icons & tint
        self._style_toolbar_button(self.btnStart, symbol="play.fill", tint="systemGreen")
        self._style_toolbar_button(self.btnDry, symbol="tortoise.fill", tint="systemTeal")
        self._style_toolbar_button(self.btnPerm, symbol="lock.shield", tint="labelColor")
        self._style_toolbar_button(self.btnMove, symbol="arrow.up.right.and.arrow.down.left", tint="labelColor")
        self._style_toolbar_button(self.btnCal, symbol="ruler", tint="labelColor")
        self._style_toolbar_button(self.btnReload, symbol="arrow.clockwise", tint="labelColor")
        self._style_toolbar_button(self.btnStop, symbol="stop.fill", tint="systemRed")

        for b in (self.btnPerm, self.btnMove, self.btnStart, self.btnDry, self.btnCal, self.btnReload, self.btnStop):
            content.addSubview_(b)

        # Segmented control for Dashboard vs Logs
        from AppKit import NSSegmentedControl
        self.segment = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
            ["Dashboard", "Logs"], 0, self, "onSegment:")
        self.segment.setFrame_(NSMakeRect(20, 430, 140, 24))
        self.segment.setSelected_forSegment_(True, 0)
        content.addSubview_(self.segment)

        # Dashboard container (not scrollable)
        self.dash = NSView.alloc().initWithFrame_(NSMakeRect(20, 20, 820, 400))
        content.addSubview_(self.dash)

        # Board preview
        from AppKit import NSImageView
        self.boardView = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 60, 280, 280))
        self.boardView.setImageScaling_(NSImageScaleProportionallyDown)
        self.dash.addSubview_(self.boardView)

        # Current word + progress
        self.currentWord = NSTextField.alloc().initWithFrame_(NSMakeRect(300, 260, 480, 40))
        self.currentWord.setBezeled_(False)
        self.currentWord.setEditable_(False)
        self.currentWord.setDrawsBackground_(False)
        self.currentWord.setFont_(NSFont.boldSystemFontOfSize_(28))
        self.currentWord.setStringValue_("")
        self.dash.addSubview_(self.currentWord)

        # Word counter label (e.g., 12 / 257)
        self.wordCounter = NSTextField.alloc().initWithFrame_(NSMakeRect(300, 235, 200, 20))
        self.wordCounter.setBezeled_(False)
        self.wordCounter.setEditable_(False)
        self.wordCounter.setDrawsBackground_(False)
        self.wordCounter.setStringValue_("0 / 0")
        self.dash.addSubview_(self.wordCounter)

        self.wordProgress = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(300, 215, 480, 16))
        try:
            self.wordProgress.setIndeterminate_(False)
            self.wordProgress.setMinValue_(0)
            self.wordProgress.setMaxValue_(100)
            self.wordProgress.setDoubleValue_(0)
        except Exception:
            pass
        self.dash.addSubview_(self.wordProgress)

        # Upcoming words
        upLabel = NSTextField.alloc().initWithFrame_(NSMakeRect(300, 190, 200, 18))
        upLabel.setBezeled_(False)
        upLabel.setEditable_(False)
        upLabel.setDrawsBackground_(False)
        upLabel.setStringValue_("Upcoming words:")
        self.dash.addSubview_(upLabel)

        # Scrollable upcoming list (separate scroller)
        self.upScroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(300, 20, 480, 160))
        self.upScroll.setHasVerticalScroller_(True)
        try:
            self.upScroll.setDrawsBackground_(False)
        except Exception:
            pass
        self.upcoming = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 480, 160))
        self.upcoming.setEditable_(False)
        self.upScroll.setDocumentView_(self.upcoming)
        self.dash.addSubview_(self.upScroll)

        # Log view (hidden by default)
        self.scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, 20, 820, 420))
        self.text = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 820, 420))
        self.text.setEditable_(False)
        self.text.setFont_(NSFont.userFixedPitchFontOfSize_(13))
        try:
            if NSColor is not None:
                self.text.setTextColor_(NSColor.labelColor())
                self.text.setBackgroundColor_(NSColor.clearColor())
        except Exception:
            pass
        self.scroll.setDocumentView_(self.text)
        self.scroll.setHasVerticalScroller_(True)
        try:
            self.scroll.setDrawsBackground_(False)
        except Exception:
            pass
        self.scroll.setHidden_(True)
        content.addSubview_(self.scroll)

        # Status label and spinner
        self.status = NSTextField.alloc().initWithFrame_(NSMakeRect(760, 420, 80, 22))
        self.status.setBezeled_(False)
        self.status.setDrawsBackground_(False)
        self.status.setEditable_(False)
        self.status.setSelectable_(False)
        self.status.setStringValue_("Idle")
        content.addSubview_(self.status)

        self.spinner = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(840, 420, 16, 16))
        try:
            from AppKit import NSProgressIndicatorStyleSpinning
            self.spinner.setStyle_(NSProgressIndicatorStyleSpinning)
        except Exception:
            self.spinner.setStyle_(1)
        self.spinner.setDisplayedWhenStopped_(False)
        content.addSubview_(self.spinner)

        self.window.makeKeyAndOrderFront_(None)
        NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

        self.logger = Logger(self.text)
        self.solver = None
        self.worker = None
        self.isRunning = False

        # Main-thread dispatcher for macOS UI calls when running background threads
        # Uses AppHelper.callAfter to execute functions on the Cocoa main thread.
        def run_on_main(func, *args, **kwargs):
            # If we're already on the main thread, run directly to avoid deadlock.
            if NSThread.isMainThread():
                return func(*args, **kwargs)
            done = threading.Event()
            result = {"val": None, "err": None}

            def _invoke():
                try:
                    result["val"] = func(*args, **kwargs)
                except BaseException as e:
                    result["err"] = e
                finally:
                    done.set()

            AppHelper.callAfter(_invoke)
            done.wait()
            if result["err"] is not None:
                raise result["err"]
            return result["val"]

        self.run_on_main = run_on_main

        self.print_welcome()
        self._start_pump()
        self._install_hotkeys()

    # --- Segmented control ---
    def onSegment_(self, sender):
        idx = sender.selectedSegment()
        self.dash.setHidden_(idx != 0)
        self.scroll.setHidden_(idx != 1)

    def _start_pump(self):
        def pump_loop():
            while True:
                self.logger.pump()
                sleep(0.08)
        t = threading.Thread(target=pump_loop, daemon=True)
        t.start()

    def _button(self, x, y, w, h, title, selector_name: str):
        b = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        b.setTitle_(title)
        b.setBezelStyle_(NSBezelStyleRounded)
        b.setTarget_(self)
        # Action must be a selector string like 'onPerm:' that maps to a Python method 'onPerm_'
        b.setAction_(selector_name)
        # Add key equivalents for quick control
        if title == "Start":
            try:
                b.setKeyEquivalent_("s")
                b.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand)
            except Exception:
                pass
        if title == "Stop":
            try:
                b.setKeyEquivalent_(".")
                b.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand)
            except Exception:
                pass
        return b

    def _style_toolbar_button(self, btn, symbol: str, tint: str = "labelColor"):
        # Apply SF Symbol icon and tint for a modern toolbar look
        try:
            if NSImage is None:
                return
            if hasattr(NSImage, 'imageWithSystemSymbolName_accessibilityDescription_'):
                img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
                if NSImageSymbolConfiguration is not None:
                    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(13.0, 0, 1)
                    img = img.imageByApplyingSymbolConfiguration_(cfg)
                btn.setImage_(img)
                btn.setImagePosition_(NSImageLeft)
                btn.setImageScaling_(NSImageScaleProportionallyDown)
            if NSColor is not None:
                color_map = {
                    'systemGreen': NSColor.systemGreenColor() if hasattr(NSColor, 'systemGreenColor') else None,
                    'systemRed': NSColor.systemRedColor() if hasattr(NSColor, 'systemRedColor') else None,
                    'systemTeal': getattr(NSColor, 'systemTealColor', lambda: NSColor.systemBlueColor())(),
                    'labelColor': NSColor.labelColor() if hasattr(NSColor, 'labelColor') else None,
                }
                tint_color = color_map.get(tint, None)
                if tint_color is not None and hasattr(btn, 'setContentTintColor_'):
                    btn.setContentTintColor_(tint_color)
        except Exception:
            pass

    def print_welcome(self):
        print("""
Welcome! Quick steps:\n\n1) Click Permissions and grant Screen Recording + Accessibility to this app.\n2) Ensure iPhone Mirroring shows WordHunt start screen.\n3) Click Move to Corner, then Start.\n4) Optional: Calibrate once; it's saved and used automatically.\n""".strip())

    def ensure_solver(self):
        if self.solver is None:
            # Disable OpenCV windows and marshal macOS UI calls to main thread
            self.solver = WordHuntSolver(no_windows=True, ui_call=self.run_on_main)
            # Subscribe to solver UI events
            self.solver.ui_event = self.onSolverEvent

    # Button actions
    def onPerm_(self, sender):
        print("Checking permissions…")
        os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenRecording"')
        os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"')
        print("If new, please restart the app after granting both.")

    def onMove_(self, sender):
        print("Moving iPhone window to corner…")
        move_iphone_mirror_to_corner()

    def _run(self, dry=False):
        try:
            self.ensure_solver()
            self.solver.no_swipe = dry
            self._set_running(True)
            self.solver.solve()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._set_running, False)

    def onStart_(self, sender):
        if self.worker and self.worker.is_alive():
            print("Already running…")
            return
        self.worker = threading.Thread(target=self._run, kwargs={'dry': False}, daemon=True)
        self.worker.start()

    def onDry_(self, sender):
        if self.worker and self.worker.is_alive():
            print("Already running…")
            return
        os.environ['WORDHUNT_DEBUG_GRID'] = '1'
        self.worker = threading.Thread(target=self._run, kwargs={'dry': True}, daemon=True)
        self.worker.start()

    def onCal_(self, sender):
        # Run calibration on a worker thread to keep UI responsive
        def _do_cal():
            try:
                self.ensure_solver()
                self.solver.calibrate(delay_seconds=3.0, show_overlay=False)
                print("Calibration saved. You can enable Grid Overlay and run Dry Run to verify.")
            except Exception as e:
                print(f"Calibration error: {e}")
        if self.worker and self.worker.is_alive():
            print("Already running…")
            return
        self.worker = threading.Thread(target=_do_cal, daemon=True)
        self.worker.start()

    def onReload_(self, sender):
        # Reload model in background; solver guards with a lock
        def _do_reload():
            try:
                self.ensure_solver()
                self.solver.reload_model()
            except Exception as e:
                print(f"Model reload error: {e}")
        t = threading.Thread(target=_do_reload, daemon=True)
        t.start()

    def onStop_(self, sender):
        if self.solver:
            self.solver.stop()
            print("Stop requested.")

    # --- UI helpers ---
    def _set_running(self, running: bool):
        self.isRunning = running
        self.btnStart.setEnabled_(not running)
        self.btnDry.setEnabled_(not running)
        self.btnCal.setEnabled_(not running)
        self.btnPerm.setEnabled_(not running)
        self.btnMove.setEnabled_(not running)
        self.btnReload.setEnabled_(True)
        self.btnStop.setEnabled_(True)
        self.status.setStringValue_("Running" if running else "Idle")
        if running:
            self.spinner.startAnimation_(None)
        else:
            self.spinner.stopAnimation_(None)

    def _install_hotkeys(self):
        try:
            from AppKit import NSEvent, NSEventMaskKeyDown
            # Global stop hotkey: Esc or Command-.
            def handler(event):
                try:
                    chars = event.charactersIgnoringModifiers()
                    if chars in ("\x1b", ".", "q", "Q"):
                        self.onStop_(None)
                except Exception:
                    pass
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, handler)
        except Exception:
            pass

    # --- Solver UI events ---
    def onSolverEvent(self, event: str, payload: dict):
        if event == 'board' and 'image' in payload:
            try:
                nsimg = self._nsimage_from_np(payload['image'])
                self.boardView.setImage_(nsimg)
            except Exception:
                pass
        elif event == 'words_list' and 'words' in payload:
            words = payload['words']
            preview_lines = []
            limit = 200
            for idx, w in enumerate(words[:limit], 1):
                preview_lines.append(f"{idx:>3}. {w}")
            preview = "\n".join(preview_lines)
            self.upcoming.setString_(preview)
            try:
                self.wordProgress.setMinValue_(0)
                self.wordProgress.setMaxValue_(max(1, len(words)))
                self.wordProgress.setDoubleValue_(0)
                self.wordCounter.setStringValue_(f"0 / {len(words)}")
            except Exception:
                pass
        elif event == 'progress':
            cur = payload.get('current', '')
            idx = int(payload.get('index', 0))
            total = int(payload.get('total', 1))
            self.currentWord.setStringValue_(cur.upper())
            try:
                # animate progress step
                self.wordProgress.incrementBy_(1.0)
                self.wordCounter.setStringValue_(f"{idx} / {total}")
            except Exception:
                pass

    def _nsimage_from_np(self, img_np):
        # Convert BGR/ARGB numpy to RGB for display, via PIL -> TIFF
        if img_np is None:
            return None
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            # drop alpha to RGB
            rgb = img_np[:, :, :3][:, :, ::-1]
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            rgb = img_np[:, :, ::-1]
        else:
            # grayscale to RGB
            rgb = np.stack([img_np]*3, axis=-1)
        pil = PILImage.fromarray(rgb)
        buf = BytesIO()
        pil.save(buf, format='TIFF')
        data = buf.getvalue()
        from Foundation import NSData
        nsdata = NSData.dataWithBytes_length_(data, len(data))
        return NSImage.alloc().initWithData_(nsdata)


if __name__ == '__main__':
    app = App()
    AppHelper.runEventLoop()
