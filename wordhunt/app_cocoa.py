import sys
import os
import threading
import queue
from time import sleep

from Cocoa import (
    NSApplication, NSApp, NSWindow, NSButton, NSTextView, NSScrollView,
    NSMakeRect, NSApplicationActivationPolicyRegular,
    NSRunningApplication, NSApplicationActivateIgnoringOtherApps,
    NSBezelStyleRounded, NSFont, NSTextField,
    NSThread, NSProgressIndicator, NSView, NSStackView,
    NSLayoutConstraint, NSLayoutRelationEqual, NSLayoutAttributeLeading,
    NSLayoutAttributeTrailing, NSLayoutAttributeTop, NSLayoutAttributeBottom,
    NSLayoutAttributeWidth, NSLayoutAttributeHeight, NSLayoutAttributeNotAnAttribute,
    NSLayoutAttributeCenterX, NSLayoutAttributeCenterY, NSLayoutRelationGreaterThanOrEqual,
    NSLayoutPriorityDefaultLow, NSLayoutPriorityDefaultHigh, NSLayoutPriorityRequired,
    NSZeroRect, NSMakeSize, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable, NSBackingStoreBuffered
)
try:
    from AppKit import (
        NSVisualEffectView, NSAppearance, NSColor, NSImage, NSImageView,
        NSImageScaleProportionallyDown, NSImageLeft, NSImageSymbolConfiguration,
        NSVisualEffectMaterialUnderWindowBackground, NSVisualEffectMaterialContentBackground,
        NSVisualEffectMaterialHUDWindow, NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectBlendingModeWithinWindow, NSVisualEffectStateActive,
        NSEventModifierFlagCommand, NSProgressIndicatorStyleSpinning,
        NSSegmentedControl, NSUserInterfaceLayoutOrientationHorizontal,
        NSUserInterfaceLayoutOrientationVertical, NSStackViewDistributionFill,
        NSStackViewDistributionFillEqually, NSBorderlessWindowMask
    )
    # try to import alignment constants, fallback to numeric values
    try:
        from AppKit import NSStackViewAlignmentCenterY, NSStackViewAlignmentLeading, NSStackViewAlignmentTop
    except ImportError:
        # define fallback values for alignment constants
        NSStackViewAlignmentTop = 1
        NSStackViewAlignmentLeading = 1
        NSStackViewAlignmentCenterY = 3
except ImportError as e:
    print(f"AppKit import error: {e}")
    # set all to None as fallback
    NSVisualEffectView = None
    NSAppearance = None
    NSColor = None
    NSImage = None
    NSImageView = None
    NSImageScaleProportionallyDown = 0
    NSImageLeft = 2
    NSImageSymbolConfiguration = None
    NSVisualEffectMaterialUnderWindowBackground = 0
    NSVisualEffectMaterialContentBackground = 0
    NSVisualEffectMaterialHUDWindow = 0
    NSVisualEffectBlendingModeBehindWindow = 0
    NSVisualEffectBlendingModeWithinWindow = 1
    NSVisualEffectStateActive = 1
    NSEventModifierFlagCommand = 1 << 20
    NSProgressIndicatorStyleSpinning = 1
    NSSegmentedControl = None
    NSUserInterfaceLayoutOrientationHorizontal = 0
    NSUserInterfaceLayoutOrientationVertical = 1
    NSStackViewDistributionFill = 0
    NSStackViewDistributionFillEqually = 2
    NSStackViewAlignmentTop = 1
    NSStackViewAlignmentLeading = 1
    NSStackViewAlignmentCenterY = 3
    NSBorderlessWindowMask = 0
from PyObjCTools import AppHelper
from io import BytesIO
from PIL import Image as PILImage
import numpy as np
import objc

from wordhunt import WordHuntSolver, move_iphone_mirror_to_corner


class Logger:
    def __init__(self, textview: NSTextView):
        self.textview = textview
        self.q = queue.Queue()
        self._orig = sys.stdout
        sys.stdout = self
        # mirror logs to file when launched from Finder
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
        # drain queue and append on main thread
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
            _append()


class App:
    def __init__(self):
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        # create modern window with proper style mask
        styleMask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | 
                     NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100.0, 100.0, 920.0, 600.0),
            styleMask,
            NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("WordHunt Solver")
        self.window.setMinSize_(NSMakeSize(760.0, 500.0))
        
        # enable full-size content view for modern look
        self.window.setTitlebarAppearsTransparent_(False)
        
        content = self.window.contentView()

        # set dark aqua appearance
        self.window.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))

        # create vibrancy background - under window background material
        self.bgEffect = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        self.bgEffect.setMaterial_(NSVisualEffectMaterialUnderWindowBackground)
        self.bgEffect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.bgEffect.setState_(NSVisualEffectStateActive)
        self.bgEffect.setAutoresizingMask_(18)  # width + height resizable
        content.addSubview_(self.bgEffect)

        # root container with proper margins
        self.rootStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.rootStack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.rootStack.setSpacing_(20.0)
        self.rootStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        content.addSubview_(self.rootStack)

        # pin root stack with standard macOS margins (20pt)
        margin = 20.0
        NSLayoutConstraint.activateConstraints_([
            self.rootStack.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), margin),
            self.rootStack.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -margin),
            self.rootStack.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), margin),
            self.rootStack.bottomAnchor().constraintEqualToAnchor_constant_(content.bottomAnchor(), -margin)
        ])

        # --- Toolbar section (no card, just buttons on transparent background) ---
        self.toolbarStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.toolbarStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.toolbarStack.setSpacing_(8.0)
        self.toolbarStack.setAlignment_(NSStackViewAlignmentCenterY)
        self.toolbarStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.rootStack.addArrangedSubview_(self.toolbarStack)

        # create buttons with proper styling
        self.btnStart = self._button("Start", "onStart:", isPrimary=True)
        self.btnDry = self._button("Dry Run", "onDry:")
        self.btnStop = self._button("Stop", "onStop:")
        self.btnCal = self._button("Calibrate", "onCal:")
        self.btnReload = self._button("Reload Model", "onReload:")
        self.btnPerm = self._button("Permissions", "onPerm:")
        self.btnMove = self._button("Move to Corner", "onMove:")

        for btn in (self.btnStart, self.btnDry, self.btnStop):
            self.toolbarStack.addArrangedSubview_(btn)
        
        # spacer to push remaining buttons to the right
        spacer = NSView.alloc().initWithFrame_(NSZeroRect)
        spacer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        spacer.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)
        self.toolbarStack.addArrangedSubview_(spacer)

        for btn in (self.btnCal, self.btnReload, self.btnPerm, self.btnMove):
            self.toolbarStack.addArrangedSubview_(btn)

        # add icons
        self._style_button(self.btnStart, "play.fill", NSColor.systemGreenColor())
        self._style_button(self.btnDry, "hare.fill", NSColor.systemTealColor())
        self._style_button(self.btnStop, "stop.fill", NSColor.systemRedColor())
        self._style_button(self.btnCal, "ruler", NSColor.labelColor())
        self._style_button(self.btnReload, "arrow.clockwise", NSColor.labelColor())
        self._style_button(self.btnPerm, "lock.shield", NSColor.labelColor())
        self._style_button(self.btnMove, "arrow.up.left.and.arrow.down.right", NSColor.labelColor())

        # --- Tab control and status row ---
        self.tabStatusRow = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.tabStatusRow.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.tabStatusRow.setAlignment_(NSStackViewAlignmentCenterY)
        self.tabStatusRow.setSpacing_(12.0)
        self.rootStack.addArrangedSubview_(self.tabStatusRow)

        self.segment = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
            ["Dashboard", "Logs"], 0, self, "onSegment:")
        self.segment.setSelected_forSegment_(True, 0)
        self.segment.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.tabStatusRow.addArrangedSubview_(self.segment)

        # spacer
        statusSpacer = NSView.alloc().initWithFrame_(NSZeroRect)
        statusSpacer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        statusSpacer.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)
        self.tabStatusRow.addArrangedSubview_(statusSpacer)

        # status indicator
        self.statusStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.statusStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.statusStack.setSpacing_(8.0)
        self.statusStack.setAlignment_(NSStackViewAlignmentCenterY)
        self.tabStatusRow.addArrangedSubview_(self.statusStack)

        self.status = NSTextField.alloc().initWithFrame_(NSZeroRect)
        self.status.setBezeled_(False)
        self.status.setDrawsBackground_(False)
        self.status.setEditable_(False)
        self.status.setSelectable_(False)
        self.status.setStringValue_("Idle")
        self.status.setFont_(NSFont.systemFontOfSize_(13.0))
        self.status.setTextColor_(NSColor.secondaryLabelColor())
        self.statusStack.addArrangedSubview_(self.status)

        self.spinner = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(0, 0, 16, 16))
        self.spinner.setStyle_(NSProgressIndicatorStyleSpinning)
        self.spinner.setControlSize_(1)  # small
        self.spinner.setDisplayedWhenStopped_(False)
        self.spinner.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.statusStack.addArrangedSubview_(self.spinner)

        # --- Content container (stretches to fill) ---
        self.contentContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        self.contentContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.rootStack.addArrangedSubview_(self.contentContainer)
        self.contentContainer.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationVertical)

        # dashboard view - use plain container with explicit constraints
        self.dashView = NSView.alloc().initWithFrame_(NSZeroRect)
        self.dashView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.contentContainer.addSubview_(self.dashView)

        NSLayoutConstraint.activateConstraints_([
            self.dashView.leadingAnchor().constraintEqualToAnchor_(self.contentContainer.leadingAnchor()),
            self.dashView.trailingAnchor().constraintEqualToAnchor_(self.contentContainer.trailingAnchor()),
            self.dashView.topAnchor().constraintEqualToAnchor_(self.contentContainer.topAnchor()),
            self.dashView.bottomAnchor().constraintEqualToAnchor_(self.contentContainer.bottomAnchor())
        ])

        # --- LEFT: Board preview (square, fixed size for now) ---
        self.boardView = NSImageView.alloc().initWithFrame_(NSZeroRect)
        self.boardView.setImageScaling_(NSImageScaleProportionallyDown)
        self.boardView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.boardView.setWantsLayer_(True)
        if self.boardView.layer():
            self.boardView.layer().setCornerRadius_(12.0)
            self.boardView.layer().setMasksToBounds_(True)
            self.boardView.layer().setBorderWidth_(0.5)
            self.boardView.layer().setBorderColor_(NSColor.separatorColor().CGColor())
        self.dashView.addSubview_(self.boardView)

        # board constraints - square, fixed on left side
        NSLayoutConstraint.activateConstraints_([
            self.boardView.widthAnchor().constraintEqualToConstant_(400.0),
            self.boardView.heightAnchor().constraintEqualToConstant_(400.0),
            self.boardView.leadingAnchor().constraintEqualToAnchor_(self.dashView.leadingAnchor()),
            self.boardView.topAnchor().constraintEqualToAnchor_(self.dashView.topAnchor())
        ])
        # prevent board from growing
        self.boardView.setContentHuggingPriority_forOrientation_(NSLayoutPriorityRequired, NSUserInterfaceLayoutOrientationHorizontal)
        self.boardView.setContentHuggingPriority_forOrientation_(NSLayoutPriorityRequired, NSUserInterfaceLayoutOrientationVertical)

        # --- RIGHT: Info panel (fills remaining space) ---
        self.infoStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.infoStack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.infoStack.setSpacing_(16.0)
        self.infoStack.setAlignment_(NSStackViewAlignmentLeading)
        self.infoStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.dashView.addSubview_(self.infoStack)

        # position info stack to the right of the board
        NSLayoutConstraint.activateConstraints_([
            self.infoStack.leadingAnchor().constraintEqualToAnchor_constant_(self.boardView.trailingAnchor(), 24.0),
            self.infoStack.trailingAnchor().constraintEqualToAnchor_(self.dashView.trailingAnchor()),
            self.infoStack.topAnchor().constraintEqualToAnchor_(self.dashView.topAnchor()),
            self.infoStack.bottomAnchor().constraintLessThanOrEqualToAnchor_(self.dashView.bottomAnchor())
        ])
        # let info stack fill remaining horizontal space
        self.infoStack.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)

        # current word display (prominent)
        self.currentWord = NSTextField.alloc().initWithFrame_(NSZeroRect)
        self.currentWord.setBezeled_(False)
        self.currentWord.setEditable_(False)
        self.currentWord.setDrawsBackground_(False)
        self.currentWord.setFont_(NSFont.systemFontOfSize_weight_(48.0, 0.4))  # bold
        self.currentWord.setTextColor_(NSColor.labelColor())
        self.currentWord.setStringValue_("—")
        self.currentWord.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.infoStack.addArrangedSubview_(self.currentWord)

        # stats row (compact)
        self.statsRow = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.statsRow.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.statsRow.setSpacing_(6.0)
        self.statsRow.setAlignment_(NSStackViewAlignmentLeading)
        self.infoStack.addArrangedSubview_(self.statsRow)

        self.wordCounter = self._label("0 / 0 words", 17.0, NSColor.labelColor())
        self.wordCounter.setFont_(NSFont.systemFontOfSize_weight_(17.0, 0.2))  # medium
        self.statsRow.addArrangedSubview_(self.wordCounter)

        self.totalScore = self._label("Potential: 0 points", 15.0, NSColor.secondaryLabelColor())
        self.statsRow.addArrangedSubview_(self.totalScore)

        # progress bar (sleeker)
        self.wordProgress = NSProgressIndicator.alloc().initWithFrame_(NSZeroRect)
        self.wordProgress.setIndeterminate_(False)
        self.wordProgress.setMinValue_(0.0)
        self.wordProgress.setMaxValue_(1.0)
        self.wordProgress.setDoubleValue_(0.0)
        self.wordProgress.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.infoStack.addArrangedSubview_(self.wordProgress)

        NSLayoutConstraint.activateConstraints_([
            self.wordProgress.heightAnchor().constraintEqualToConstant_(6.0)
        ])

        # separator line
        separator = NSView.alloc().initWithFrame_(NSZeroRect)
        separator.setTranslatesAutoresizingMaskIntoConstraints_(False)
        separator.setWantsLayer_(True)
        if separator.layer():
            separator.layer().setBackgroundColor_(NSColor.separatorColor().CGColor())
        self.infoStack.addArrangedSubview_(separator)
        NSLayoutConstraint.activateConstraints_([
            separator.heightAnchor().constraintEqualToConstant_(1.0)
        ])

        # upcoming words label
        upcomingLabel = self._label("Upcoming Words", 12.0, NSColor.tertiaryLabelColor())
        upcomingLabel.setFont_(NSFont.boldSystemFontOfSize_(12.0))
        self.infoStack.addArrangedSubview_(upcomingLabel)

        # scrollable upcoming words list (clean, borderless)
        self.upScroll = NSScrollView.alloc().initWithFrame_(NSZeroRect)
        self.upScroll.setHasVerticalScroller_(True)
        self.upScroll.setDrawsBackground_(False)
        self.upScroll.setBorderType_(0)  # no border
        self.upScroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.infoStack.addArrangedSubview_(self.upScroll)

        # make scroll view stretch to fill remaining space
        self.upScroll.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationVertical)

        self.upcoming = NSTextView.alloc().initWithFrame_(NSZeroRect)
        self.upcoming.setEditable_(False)
        self.upcoming.setFont_(NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0))
        self.upcoming.setTextColor_(NSColor.secondaryLabelColor())
        self.upcoming.setBackgroundColor_(NSColor.clearColor())
        self.upcoming.setTextContainerInset_(NSMakeSize(8.0, 8.0))
        self.upcoming.setMinSize_(NSMakeSize(0.0, 0.0))
        self.upcoming.setMaxSize_(NSMakeSize(1e7, 1e7))
        self.upcoming.setVerticallyResizable_(True)
        self.upcoming.setHorizontallyResizable_(False)
        self.upcoming.textContainer().setWidthTracksTextView_(True)
        self.upScroll.setDocumentView_(self.upcoming)

        # logs view (initially hidden, clean design)
        self.logsScroll = NSScrollView.alloc().initWithFrame_(NSZeroRect)
        self.logsScroll.setHasVerticalScroller_(True)
        self.logsScroll.setDrawsBackground_(False)
        self.logsScroll.setBorderType_(0)
        self.logsScroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.logsScroll.setHidden_(True)
        self.contentContainer.addSubview_(self.logsScroll)

        NSLayoutConstraint.activateConstraints_([
            self.logsScroll.leadingAnchor().constraintEqualToAnchor_(self.contentContainer.leadingAnchor()),
            self.logsScroll.trailingAnchor().constraintEqualToAnchor_(self.contentContainer.trailingAnchor()),
            self.logsScroll.topAnchor().constraintEqualToAnchor_(self.contentContainer.topAnchor()),
            self.logsScroll.bottomAnchor().constraintEqualToAnchor_(self.contentContainer.bottomAnchor())
        ])

        self.text = NSTextView.alloc().initWithFrame_(NSZeroRect)
        self.text.setEditable_(False)
        self.text.setFont_(NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0))
        self.text.setTextColor_(NSColor.labelColor())
        self.text.setBackgroundColor_(NSColor.clearColor())
        self.text.setTextContainerInset_(NSMakeSize(12.0, 12.0))
        self.text.setMinSize_(NSMakeSize(0.0, 0.0))
        self.text.setMaxSize_(NSMakeSize(1e7, 1e7))
        self.text.setVerticallyResizable_(True)
        self.text.setHorizontallyResizable_(False)
        self.text.textContainer().setWidthTracksTextView_(True)
        self.logsScroll.setDocumentView_(self.text)

        self.window.makeKeyAndOrderFront_(None)
        NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

        self.logger = Logger(self.text)
        self.solver = None
        self.worker = None
        self.isRunning = False

        # main-thread dispatcher
        def run_on_main(func, *args, **kwargs):
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


    def _button(self, title: str, selector_name: str, isPrimary: bool = False):
        """create a properly styled button"""
        btn = NSButton.alloc().initWithFrame_(NSZeroRect)
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setTarget_(self)
        btn.setAction_(selector_name)
        btn.setTranslatesAutoresizingMaskIntoConstraints_(False)
        btn.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultHigh, NSUserInterfaceLayoutOrientationHorizontal)
        
        if isPrimary:
            # make primary button more prominent
            btn.setKeyEquivalent_("\r")  # return key
        
        # keyboard shortcuts
        if title == "Start":
            btn.setKeyEquivalent_("s")
            btn.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand)
        elif title == "Stop":
            btn.setKeyEquivalent_(".")
            btn.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand)
        
        return btn

    def _label(self, text: str, size: float, color):
        """create a label"""
        lbl = NSTextField.alloc().initWithFrame_(NSZeroRect)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setStringValue_(text)
        lbl.setFont_(NSFont.systemFontOfSize_(size))
        lbl.setTextColor_(color)
        lbl.setTranslatesAutoresizingMaskIntoConstraints_(False)
        return lbl

    def _style_button(self, btn, symbol: str, color):
        """add SF Symbol and color to button"""
        try:
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
            if img:
                cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(13.0, 0, 1)
                img = img.imageByApplyingSymbolConfiguration_(cfg)
                btn.setImage_(img)
                btn.setImagePosition_(NSImageLeft)
                if hasattr(btn, 'setContentTintColor_'):
                    btn.setContentTintColor_(color)
        except Exception:
            pass

    def onSegment_(self, sender):
        """switch between dashboard and logs"""
        idx = sender.selectedSegment()
        self.dashView.setHidden_(idx != 0)
        self.logsScroll.setHidden_(idx != 1)
        if idx == 1:
            try:
                self.logger.pump()
            except Exception:
                pass

    def _start_pump(self):
        """start log pumping thread"""
        def pump_loop():
            while True:
                self.logger.pump()
                sleep(0.08)
        t = threading.Thread(target=pump_loop, daemon=True)
        t.start()

    def print_welcome(self):
        print("""Welcome to WordHunt Solver!

Quick Start:
1. Click 'Permissions' and grant Screen Recording + Accessibility
2. Open iPhone Mirroring with WordHunt start screen
3. Click 'Move to Corner', then 'Start'
4. (Optional) Run 'Calibrate' once for better accuracy

Keyboard Shortcuts:
⌘S - Start solving
⌘. - Stop
        """.strip())

    def ensure_solver(self):
        if self.solver is None:
            self.solver = WordHuntSolver(no_windows=True, ui_call=self.run_on_main)
            self.solver.ui_event = self.onSolverEvent

    # button actions
    def onPerm_(self, sender):
        print("Opening System Preferences...")
        os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenRecording"')
        os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"')
        print("⚠️  Please grant both permissions and restart the app")

    def onMove_(self, sender):
        print("Moving iPhone Mirroring window to corner...")
        move_iphone_mirror_to_corner()

    def _run(self, dry=False):
        try:
            self.ensure_solver()
            self.solver.no_swipe = dry
            mode = "Dry Run" if dry else "Solving"
            self._set_running(True, mode)
            self.solver.solve()
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._set_running, False)

    def onStart_(self, sender):
        if self.worker and self.worker.is_alive():
            print("⚠️  Already running...")
            return
        self.worker = threading.Thread(target=self._run, kwargs={'dry': False}, daemon=True)
        self.worker.start()

    def onDry_(self, sender):
        if self.worker and self.worker.is_alive():
            print("⚠️  Already running...")
            return
        os.environ['WORDHUNT_DEBUG_GRID'] = '1'
        self.worker = threading.Thread(target=self._run, kwargs={'dry': True}, daemon=True)
        self.worker.start()

    def onCal_(self, sender):
        def _do_cal():
            try:
                self.ensure_solver()
                print("🎯 Starting calibration in 3 seconds...")
                self.solver.calibrate(delay_seconds=3.0, show_overlay=False)
                print("✅ Calibration saved successfully")
            except Exception as e:
                print(f"❌ Calibration error: {e}")
        if self.worker and self.worker.is_alive():
            print("⚠️  Already running...")
            return
        self.worker = threading.Thread(target=_do_cal, daemon=True)
        self.worker.start()

    def onReload_(self, sender):
        def _do_reload():
            try:
                self.ensure_solver()
                print("🔄 Reloading model...")
                self.solver.reload_model()
                print("✅ Model reloaded successfully!")
            except Exception as e:
                print(f"❌ Model reload error: {e}")
            finally:
                # restore status on main thread
                try:
                    from PyObjCTools import AppHelper
                    AppHelper.callAfter(self.status.setStringValue_, "Idle")
                    AppHelper.callAfter(self.spinner.stopAnimation_, None)
                except Exception:
                    self.status.setStringValue_("Idle")
                    self.spinner.stopAnimation_(None)
        # show immediate feedback
        print("Starting model reload...")
        self.status.setStringValue_("Reloading model…")
        self.spinner.startAnimation_(None)
        # ensure any buffered output appears quickly
        try:
            self.logger.pump()
        except Exception:
            pass
        t = threading.Thread(target=_do_reload, daemon=True)
        t.start()

    def onStop_(self, sender):
        if self.solver:
            self.solver.stop()
            print("🛑 Stop requested")

    def _set_running(self, running: bool, mode: str = None):
        """update UI for running state"""
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
        """install global hotkeys"""
        try:
            from AppKit import NSEvent, NSEventMaskKeyDown
            def handler(event):
                try:
                    chars = event.charactersIgnoringModifiers()
                    if chars in ("\x1b", "."):  # Esc or .
                        self.onStop_(None)
                except Exception:
                    pass
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, handler)
        except Exception:
            pass

    # solver UI events
    def onSolverEvent(self, event: str, payload: dict):
        """handle solver UI updates"""
        if event == 'board' and 'image' in payload:
            try:
                nsimg = self._nsimage_from_np(payload['image'])
                if nsimg:
                    self.boardView.setImage_(nsimg)
            except Exception as e:
                print(f"Error updating board: {e}")
        
        elif event == 'words_list' and 'words' in payload:
            words = payload['words']
            lines = []
            for idx, w in enumerate(words[:200], 1):
                lines.append(f"{idx:>3}. {w.upper()}")
            self.upcoming.setString_("\n".join(lines))
            
            self.wordProgress.setMinValue_(0)
            self.wordProgress.setMaxValue_(max(1, len(words)))
            self.wordProgress.setDoubleValue_(0)
            word_label = "word" if len(words) == 1 else "words"
            self.wordCounter.setStringValue_(f"0 / {len(words)} {word_label}")
            self.currentWord.setStringValue_("—")
        
        elif event == 'potential_score':
            score = payload.get('score', 0)
            point_label = "point" if score == 1 else "points"
            self.totalScore.setStringValue_(f"Potential: {score:,} {point_label}")
        
        elif event == 'progress':
            cur = payload.get('current', '')
            idx = int(payload.get('index', 0))
            total = int(payload.get('total', 1))
            self.currentWord.setStringValue_(cur.upper())
            self.wordProgress.setDoubleValue_(idx)
            word_label = "word" if total == 1 else "words"
            self.wordCounter.setStringValue_(f"{idx} / {total} {word_label}")

    def _nsimage_from_np(self, img_np):
        """convert numpy array to NSImage"""
        if img_np is None:
            return None
        try:
            # convert to RGB
            if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                rgb = img_np[:, :, :3][:, :, ::-1]
            elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
                rgb = img_np[:, :, ::-1]
            else:
                rgb = np.stack([img_np]*3, axis=-1)
            
            pil = PILImage.fromarray(rgb)
            buf = BytesIO()
            pil.save(buf, format='PNG')
            data = buf.getvalue()
            
            from Foundation import NSData
            nsdata = NSData.dataWithBytes_length_(data, len(data))
            return NSImage.alloc().initWithData_(nsdata)
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None


if __name__ == '__main__':
    app = App()
    AppHelper.runEventLoop()
