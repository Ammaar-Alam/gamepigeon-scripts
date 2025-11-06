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
    NSThread, NSProgressIndicator, NSView, NSStackView,
    NSLayoutConstraint, NSLayoutRelationEqual, NSLayoutAttributeLeading,
    NSLayoutAttributeTrailing, NSLayoutAttributeTop, NSLayoutAttributeBottom,
    NSLayoutAttributeWidth, NSLayoutAttributeHeight, NSLayoutAttributeNotAnAttribute,
    NSLayoutPriorityDefaultLow, NSZeroRect
)
try:
    # AppKit-only conveniences
    from AppKit import (
        NSVisualEffectView, NSAppearance, NSAppearanceNameDarkAqua, NSColor, NSImage,
        NSImageScaleProportionallyDown, NSImageLeft, NSImageSymbolConfiguration,
        NSVisualEffectMaterialSidebar, NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectStateActive, NSEventModifierFlagCommand, NSProgressIndicatorStyleSpinning,
        NSSegmentedControl, NSImageView,
        NSUserInterfaceLayoutOrientationHorizontal, NSUserInterfaceLayoutOrientationVertical,
        NSStackViewDistributionEqualSpacing, NSStackViewDistributionFillProportionally,
        NSStackViewAlignmentCenterY, NSStackViewAlignmentLeading
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
    NSSegmentedControl = None
    NSImageView = None
    NSProgressIndicatorStyleSpinning = 1
    NSUserInterfaceLayoutOrientationHorizontal = 0
    NSUserInterfaceLayoutOrientationVertical = 1
    NSStackViewDistributionEqualSpacing = 2
    NSStackViewDistributionFillProportionally = 1
    NSStackViewAlignmentCenterY = 3
    NSStackViewAlignmentLeading = 1
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
        try:
            self.window.setMinSize_((760.0, 500.0))
        except Exception:
            pass

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

        content.setWantsLayer_(True)

        # Root vertical stack providing adaptive layout
        self.rootStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.rootStack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.rootStack.setSpacing_(18.0)
        self.rootStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        content.addSubview_(self.rootStack)

        inset = 24.0
        constraints = [
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.rootStack, NSLayoutAttributeLeading, NSLayoutRelationEqual, content, NSLayoutAttributeLeading, 1.0, inset),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.rootStack, NSLayoutAttributeTrailing, NSLayoutRelationEqual, content, NSLayoutAttributeTrailing, 1.0, -inset),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.rootStack, NSLayoutAttributeTop, NSLayoutRelationEqual, content, NSLayoutAttributeTop, 1.0, inset),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.rootStack, NSLayoutAttributeBottom, NSLayoutRelationEqual, content, NSLayoutAttributeBottom, 1.0, -inset)
        ]
        content.addConstraints_(constraints)

        # --- Toolbar buttons ---
        self.controlsStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.controlsStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.controlsStack.setSpacing_(12.0)
        self.controlsStack.setDistribution_(NSStackViewDistributionFillProportionally)
        self.controlsStack.setAlignment_(NSStackViewAlignmentCenterY)
        self.rootStack.addArrangedSubview_(self.controlsStack)

        self.btnPerm   = self._button("Permissions", "onPerm:")
        self.btnMove   = self._button("Move to Corner", "onMove:")
        self.btnStart  = self._button("Start", "onStart:")
        self.btnDry    = self._button("Dry Run", "onDry:")
        self.btnCal    = self._button("Calibrate", "onCal:")
        self.btnReload = self._button("Reload Model", "onReload:")
        self.btnStop   = self._button("Stop", "onStop:")

        for b in (self.btnPerm, self.btnMove, self.btnStart, self.btnDry, self.btnCal, self.btnReload, self.btnStop):
            self.controlsStack.addArrangedSubview_(b)

        # Icons & tint styling
        self._style_toolbar_button(self.btnStart, symbol="play.fill", tint="systemGreen")
        self._style_toolbar_button(self.btnDry, symbol="tortoise.fill", tint="systemTeal")
        self._style_toolbar_button(self.btnPerm, symbol="lock.shield", tint="labelColor")
        self._style_toolbar_button(self.btnMove, symbol="arrow.up.right.and.arrow.down.left", tint="labelColor")
        self._style_toolbar_button(self.btnCal, symbol="ruler", tint="labelColor")
        self._style_toolbar_button(self.btnReload, symbol="arrow.clockwise", tint="labelColor")
        self._style_toolbar_button(self.btnStop, symbol="stop.fill", tint="systemRed")

        # --- Tab row: dashboard/logs + status ---
        self.tabRow = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.tabRow.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.tabRow.setAlignment_(NSStackViewAlignmentCenterY)
        self.tabRow.setSpacing_(12.0)
        self.rootStack.addArrangedSubview_(self.tabRow)

        if NSSegmentedControl is not None:
            self.segment = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
                ["Dashboard", "Logs"], 0, self, "onSegment:")
            self.segment.setSelected_forSegment_(True, 0)
        else:
            self.segment = self._button("Dashboard", "onSegment:")
        self.tabRow.addArrangedSubview_(self.segment)
        try:
            self.segment.setHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow + 5, NSUserInterfaceLayoutOrientationHorizontal)
        except Exception:
            pass

        # Flexible spacer so status aligns right
        self._spacer = NSView.alloc().initWithFrame_(NSZeroRect)
        self._spacer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.tabRow.addArrangedSubview_(self._spacer)
        try:
            self._spacer.setHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow, NSUserInterfaceLayoutOrientationHorizontal)
        except Exception:
            pass

        self.statusStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.statusStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.statusStack.setSpacing_(6.0)
        self.tabRow.addArrangedSubview_(self.statusStack)

        self.status = NSTextField.alloc().initWithFrame_(NSZeroRect)
        self.status.setBezeled_(False)
        self.status.setDrawsBackground_(False)
        self.status.setEditable_(False)
        self.status.setSelectable_(False)
        self.status.setStringValue_("Idle")
        try:
            if NSColor is not None and hasattr(NSColor, 'secondaryLabelColor'):
                self.status.setTextColor_(NSColor.secondaryLabelColor())
        except Exception:
            pass
        self.statusStack.addArrangedSubview_(self.status)

        self.spinner = NSProgressIndicator.alloc().initWithFrame_(NSZeroRect)
        try:
            self.spinner.setStyle_(NSProgressIndicatorStyleSpinning)
        except Exception:
            self.spinner.setStyle_(1)
        self.spinner.setDisplayedWhenStopped_(False)
        self.statusStack.addArrangedSubview_(self.spinner)

        # Ensure the content area stretches
        self.contentContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        self.contentContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        try:
            self.contentContainer.setHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow, NSUserInterfaceLayoutOrientationVertical)
        except Exception:
            pass
        self.rootStack.addArrangedSubview_(self.contentContainer)

        # Dashboard and logs occupy the same container (toggled hidden)
        self.dash = NSView.alloc().initWithFrame_(NSZeroRect)
        self.dash.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.contentContainer.addSubview_(self.dash)

        self.scroll = NSScrollView.alloc().initWithFrame_(NSZeroRect)
        self.scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.text = NSTextView.alloc().initWithFrame_(NSZeroRect)
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
        self.contentContainer.addSubview_(self.scroll)

        # Pin dashboards to container edges
        for view in (self.dash, self.scroll):
            constraints = [
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    view, NSLayoutAttributeLeading, NSLayoutRelationEqual, self.contentContainer, NSLayoutAttributeLeading, 1.0, 0.0),
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    view, NSLayoutAttributeTrailing, NSLayoutRelationEqual, self.contentContainer, NSLayoutAttributeTrailing, 1.0, 0.0),
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    view, NSLayoutAttributeTop, NSLayoutRelationEqual, self.contentContainer, NSLayoutAttributeTop, 1.0, 0.0),
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    view, NSLayoutAttributeBottom, NSLayoutRelationEqual, self.contentContainer, NSLayoutAttributeBottom, 1.0, 0.0)
            ]
            self.contentContainer.addConstraints_(constraints)

        # --- Dashboard layout ---
        self.dashboardStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.dashboardStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.dashboardStack.setSpacing_(24.0)
        self.dashboardStack.setAlignment_(NSStackViewAlignmentLeading)
        self.dashboardStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.dash.addSubview_(self.dashboardStack)

        dashConstraints = [
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.dashboardStack, NSLayoutAttributeLeading, NSLayoutRelationEqual, self.dash, NSLayoutAttributeLeading, 1.0, 0.0),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.dashboardStack, NSLayoutAttributeTrailing, NSLayoutRelationEqual, self.dash, NSLayoutAttributeTrailing, 1.0, 0.0),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.dashboardStack, NSLayoutAttributeTop, NSLayoutRelationEqual, self.dash, NSLayoutAttributeTop, 1.0, 0.0),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.dashboardStack, NSLayoutAttributeBottom, NSLayoutRelationEqual, self.dash, NSLayoutAttributeBottom, 1.0, 0.0)
        ]
        self.dash.addConstraints_(dashConstraints)

        if NSImageView is not None:
            self.boardView = NSImageView.alloc().initWithFrame_(NSZeroRect)
            if NSImageScaleProportionallyDown:
                self.boardView.setImageScaling_(NSImageScaleProportionallyDown)
        else:
            self.boardView = NSView.alloc().initWithFrame_(NSZeroRect)
        self.boardView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.dashboardStack.addArrangedSubview_(self.boardView)

        boardConstraints = [
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.boardView, NSLayoutAttributeWidth, NSLayoutRelationEqual, None, NSLayoutAttributeNotAnAttribute, 1.0, 280.0),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.boardView, NSLayoutAttributeHeight, NSLayoutRelationEqual, None, NSLayoutAttributeNotAnAttribute, 1.0, 280.0)
        ]
        self.boardView.addConstraints_(boardConstraints)

        self.infoStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.infoStack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.infoStack.setSpacing_(12.0)
        self.infoStack.setAlignment_(NSStackViewAlignmentLeading)
        self.dashboardStack.addArrangedSubview_(self.infoStack)

        self.currentWord = NSTextField.alloc().initWithFrame_(NSZeroRect)
        self.currentWord.setBezeled_(False)
        self.currentWord.setEditable_(False)
        self.currentWord.setDrawsBackground_(False)
        self.currentWord.setFont_(NSFont.boldSystemFontOfSize_(30))
        self.currentWord.setStringValue_("")
        self.infoStack.addArrangedSubview_(self.currentWord)

        # Progress + counts row
        self.metaRow = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.metaRow.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.metaRow.setSpacing_(12.0)
        self.infoStack.addArrangedSubview_(self.metaRow)

        self.wordCounter = self._label("0 / 0")
        try:
            self.wordCounter.setFont_(NSFont.systemFontOfSize_(15))
        except Exception:
            pass
        self.metaRow.addArrangedSubview_(self.wordCounter)

        self.totalScore = self._label("Potential: 0")
        try:
            self.totalScore.setFont_(NSFont.systemFontOfSize_(15))
        except Exception:
            pass
        self.metaRow.addArrangedSubview_(self.totalScore)

        self.wordProgress = NSProgressIndicator.alloc().initWithFrame_(NSZeroRect)
        try:
            self.wordProgress.setIndeterminate_(False)
            self.wordProgress.setMinValue_(0.0)
            self.wordProgress.setMaxValue_(1.0)
            self.wordProgress.setDoubleValue_(0.0)
            self.wordProgress.setUsesThreadedAnimation_(True)
        except Exception:
            pass
        self.infoStack.addArrangedSubview_(self.wordProgress)

        self.upcomingTitle = self._label("Upcoming words:")
        try:
            self.upcomingTitle.setFont_(NSFont.systemFontOfSize_(15))
        except Exception:
            pass
        self.infoStack.addArrangedSubview_(self.upcomingTitle)

        self.upScroll = NSScrollView.alloc().initWithFrame_(NSZeroRect)
        self.upScroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.upScroll.setHasVerticalScroller_(True)
        try:
            self.upScroll.setDrawsBackground_(False)
        except Exception:
            pass
        try:
            self.upScroll.setHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow, NSUserInterfaceLayoutOrientationVertical)
        except Exception:
            pass
        self.upcoming = NSTextView.alloc().initWithFrame_(NSZeroRect)
        self.upcoming.setEditable_(False)
        try:
            mono = getattr(NSFont, 'monospacedSystemFontOfSize_weight_', None)
            if mono is not None:
                self.upcoming.setFont_(mono(13, 0))
            if NSColor is not None:
                self.upcoming.setTextColor_(NSColor.labelColor())
                self.upcoming.setBackgroundColor_(NSColor.clearColor())
        except Exception:
            pass
        self.upScroll.setDocumentView_(self.upcoming)
        self.infoStack.addArrangedSubview_(self.upScroll)

        # Give the upcoming list a reasonable minimum height but allow expansion
        upcomingConstraints = [
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.upScroll, NSLayoutAttributeHeight, NSLayoutRelationEqual, None, NSLayoutAttributeNotAnAttribute, 1.0, 170.0)
        ]
        self.upScroll.addConstraints_(upcomingConstraints)

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
        idx = 0
        try:
            idx = sender.selectedSegment()
        except Exception:
            pass
        self.dash.setHidden_(idx != 0)
        self.scroll.setHidden_(idx != 1)

    def _start_pump(self):
        def pump_loop():
            while True:
                self.logger.pump()
                sleep(0.08)
        t = threading.Thread(target=pump_loop, daemon=True)
        t.start()

    def _button(self, title: str, selector_name: str, min_width: float = 0.0):
        b = NSButton.alloc().initWithFrame_(NSZeroRect)
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
        if min_width and min_width > 0:
            constraint = NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                b, NSLayoutAttributeWidth, NSLayoutRelationEqual, None, NSLayoutAttributeNotAnAttribute, 1.0, min_width)
            b.addConstraint_(constraint)
        return b

    def _label(self, text: str):
        lbl = NSTextField.alloc().initWithFrame_(NSZeroRect)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setStringValue_(text)
        return lbl

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
            mode = "Dry Run" if dry else "Solving"
            self._set_running(True, mode)
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
    def _set_running(self, running: bool, mode: str = None):
        self.isRunning = running
        self.btnStart.setEnabled_(not running)
        self.btnDry.setEnabled_(not running)
        self.btnCal.setEnabled_(not running)
        self.btnPerm.setEnabled_(not running)
        self.btnMove.setEnabled_(not running)
        self.btnReload.setEnabled_(True)
        self.btnStop.setEnabled_(True)
        status_text = "Idle"
        if running:
            status_text = mode if mode else "Running"
        self.status.setStringValue_(status_text)
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
                if hasattr(self.boardView, 'setImage_'):
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
            self.currentWord.setStringValue_("")
        elif event == 'potential_score':
            score = payload.get('score', 0)
            self.totalScore.setStringValue_(f"Potential: {score:,}")
        elif event == 'progress':
            cur = payload.get('current', '')
            idx = int(payload.get('index', 0))
            total = int(payload.get('total', 1))
            self.currentWord.setStringValue_(cur.upper())
            try:
                # animate progress step
                self.wordProgress.setDoubleValue_(idx)
                self.wordCounter.setStringValue_(f"{idx} / {total}")
            except Exception:
                pass

    def _nsimage_from_np(self, img_np):
        # Convert BGR/ARGB numpy to RGB for display, via PIL -> TIFF
        if img_np is None:
            return None
        if NSImage is None:
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
