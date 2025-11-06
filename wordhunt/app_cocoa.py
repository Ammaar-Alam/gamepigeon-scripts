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
from Foundation import NSMakeRange
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
            # append attributed chunk for reliable rendering on vibrancy
            try:
                from AppKit import NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName
                from Foundation import NSAttributedString
                attrs = {
                    NSFontAttributeName: NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0),
                    NSForegroundColorAttributeName: NSColor.whiteColor(),
                }
                chunk = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
                ts = self.textview.textStorage()
                ts.beginEditing()
                ts.appendAttributedString_(chunk)
                ts.endEditing()
                try:
                    self.textview.layoutManager().ensureLayoutForTextContainer_(self.textview.textContainer())
                    self.textview.setNeedsDisplay_(True)
                except Exception:
                    pass
            except Exception:
                # fallback to plain string if attributed append fails
                current = self.textview.string() or ""
                self.textview.setString_(current + text)
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
        
        # setup application menu for Command+Q to work
        self._setup_app_menu()

        # create modern window with proper style mask
        styleMask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | 
                     NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100.0, 100.0, 980.0, 640.0),  # slightly larger default
            styleMask,
            NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("WordHunt Solver")
        self.window.setMinSize_(NSMakeSize(860.0, 560.0))  # slightly larger minimum
        
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

        # pin root stack with margins (no bottom margin since contentContainer is separate)
        NSLayoutConstraint.activateConstraints_([
            self.rootStack.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), 12.0),
            self.rootStack.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -20.0),
            self.rootStack.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), 20.0)
            # no bottom constraint - contentContainer handles that space
        ])

        # --- Toolbar section with visual container ---
        toolbarContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        toolbarContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        toolbarContainer.setWantsLayer_(True)
        if toolbarContainer.layer():
            toolbarContainer.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(1.0, 0.03).CGColor())
            toolbarContainer.layer().setCornerRadius_(8.0)
        self.rootStack.addArrangedSubview_(toolbarContainer)
        
        self.toolbarStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.toolbarStack.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        self.toolbarStack.setSpacing_(8.0)
        self.toolbarStack.setAlignment_(NSStackViewAlignmentCenterY)
        self.toolbarStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        toolbarContainer.addSubview_(self.toolbarStack)
        
        # constrain toolbar stack within container with padding
        NSLayoutConstraint.activateConstraints_([
            self.toolbarStack.leadingAnchor().constraintEqualToAnchor_constant_(toolbarContainer.leadingAnchor(), 12.0),
            self.toolbarStack.trailingAnchor().constraintEqualToAnchor_constant_(toolbarContainer.trailingAnchor(), -12.0),
            self.toolbarStack.topAnchor().constraintEqualToAnchor_constant_(toolbarContainer.topAnchor(), 8.0),
            self.toolbarStack.bottomAnchor().constraintEqualToAnchor_constant_(toolbarContainer.bottomAnchor(), -8.0)
        ])

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
        
        # add vertical separator after action buttons
        sep1 = NSView.alloc().initWithFrame_(NSZeroRect)
        sep1.setTranslatesAutoresizingMaskIntoConstraints_(False)
        sep1.setWantsLayer_(True)
        if sep1.layer():
            sep1.layer().setBackgroundColor_(NSColor.separatorColor().CGColor())
        self.toolbarStack.addArrangedSubview_(sep1)
        NSLayoutConstraint.activateConstraints_([
            sep1.widthAnchor().constraintEqualToConstant_(1.0),
            sep1.heightAnchor().constraintEqualToConstant_(24.0)
        ])
        
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

        # --- Content container (direct child of window, flush with edges) ---
        self.contentContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        self.contentContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        # add directly to window content view, NOT to rootStack
        content.addSubview_(self.contentContainer)
        
        # position content container below rootStack (which contains toolbar + tabs) and flush with window edges horizontally
        NSLayoutConstraint.activateConstraints_([
            self.contentContainer.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()),  # flush left
            self.contentContainer.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -20.0),  # add right margin
            self.contentContainer.topAnchor().constraintEqualToAnchor_constant_(self.rootStack.bottomAnchor(), 8.0),  # small gap below rootStack
            self.contentContainer.bottomAnchor().constraintEqualToAnchor_constant_(content.bottomAnchor(), -20.0)  # standard bottom margin
        ])

        # dashboard view - use plain container with explicit constraints
        self.dashView = NSView.alloc().initWithFrame_(NSZeroRect)
        self.dashView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.contentContainer.addSubview_(self.dashView)

        NSLayoutConstraint.activateConstraints_([
            # pin main dashboard fully to container so it uses entire width
            self.dashView.leadingAnchor().constraintEqualToAnchor_(self.contentContainer.leadingAnchor()),
            self.dashView.trailingAnchor().constraintEqualToAnchor_(self.contentContainer.trailingAnchor()),
            self.dashView.topAnchor().constraintEqualToAnchor_(self.contentContainer.topAnchor()),
            self.dashView.bottomAnchor().constraintEqualToAnchor_(self.contentContainer.bottomAnchor())
        ])

        # --- LEFT: Board preview (square, fixed size for now) ---
        # create container for shadow (shadow needs to be outside masksToBounds)
        self.boardContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        self.boardContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.boardContainer.setWantsLayer_(True)
        if self.boardContainer.layer():
            # add shadow to container
            try:
                self.boardContainer.layer().setShadowColor_(NSColor.blackColor().CGColor())
                self.boardContainer.layer().setShadowOffset_(NSMakeSize(0.0, -2.0))
                self.boardContainer.layer().setShadowOpacity_(0.3)
                self.boardContainer.layer().setShadowRadius_(8.0)
            except Exception:
                pass
        self.dashView.addSubview_(self.boardContainer)
        
        self.boardView = NSImageView.alloc().initWithFrame_(NSZeroRect)
        # use aspect fit to prevent border overflow, and align image to center
        self.boardView.setImageScaling_(NSImageScaleProportionallyDown)  # scale down to fit, maintain aspect
        try:
            self.boardView.setImageAlignment_(2)  # NSImageAlignCenter = 2
        except Exception:
            pass  # fallback if not available
        self.boardView.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.boardView.setWantsLayer_(True)
        if self.boardView.layer():
            self.boardView.layer().setCornerRadius_(16.0)  # larger radius
            self.boardView.layer().setMasksToBounds_(True)  # clip to bounds to prevent border overflow
            # restore a subtle border for definition
            self.boardView.layer().setBorderWidth_(0.5)
            self.boardView.layer().setBorderColor_(NSColor.separatorColor().CGColor())
        self.boardContainer.addSubview_(self.boardView)

        # board container constraints - square, flush left (store width/height to adjust dynamically)
        self.boardWidthConstraint = self.boardContainer.widthAnchor().constraintEqualToConstant_(400.0)
        self.boardHeightConstraint = self.boardContainer.heightAnchor().constraintEqualToConstant_(400.0)
        NSLayoutConstraint.activateConstraints_([
            self.boardWidthConstraint,
            self.boardHeightConstraint,
            self.boardContainer.leadingAnchor().constraintEqualToAnchor_(self.dashView.leadingAnchor()),  # flush with contentContainer (which is flush with window)
            self.boardContainer.topAnchor().constraintEqualToAnchor_(self.dashView.topAnchor())
        ])
        
        # board view fills container
        NSLayoutConstraint.activateConstraints_([
            self.boardView.leadingAnchor().constraintEqualToAnchor_(self.boardContainer.leadingAnchor()),
            self.boardView.trailingAnchor().constraintEqualToAnchor_(self.boardContainer.trailingAnchor()),
            self.boardView.topAnchor().constraintEqualToAnchor_(self.boardContainer.topAnchor()),
            self.boardView.bottomAnchor().constraintEqualToAnchor_(self.boardContainer.bottomAnchor())
        ])
        # prevent board from growing
        self.boardView.setContentHuggingPriority_forOrientation_(NSLayoutPriorityRequired, NSUserInterfaceLayoutOrientationHorizontal)
        self.boardView.setContentHuggingPriority_forOrientation_(NSLayoutPriorityRequired, NSUserInterfaceLayoutOrientationVertical)

        # --- RIGHT: Info panel (fills remaining space) ---
        self.infoStack = NSStackView.alloc().initWithFrame_(NSZeroRect)
        self.infoStack.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        self.infoStack.setSpacing_(16.0)
        # Fill distribution + leading alignment for natural left alignment
        self.infoStack.setDistribution_(NSStackViewDistributionFill)
        self.infoStack.setAlignment_(NSStackViewAlignmentLeading)
        self.infoStack.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.dashView.addSubview_(self.infoStack)

        # position info stack to the right of the board with tight spacing
        NSLayoutConstraint.activateConstraints_([
            self.infoStack.leadingAnchor().constraintEqualToAnchor_constant_(self.boardContainer.trailingAnchor(), 12.0),  # tighter spacing
            self.infoStack.trailingAnchor().constraintEqualToAnchor_constant_(self.dashView.trailingAnchor(), -12.0),
            # align the top of info with the top of the captured board
            self.infoStack.topAnchor().constraintEqualToAnchor_(self.boardContainer.topAnchor()),
            self.infoStack.bottomAnchor().constraintLessThanOrEqualToAnchor_(self.dashView.bottomAnchor())
        ])
        # let info stack fill remaining horizontal space
        self.infoStack.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)
        # keep text from stretching the column width (apply after currentWord is created)

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
        # avoid long words forcing the info column to expand
        try:
            self.currentWord.setContentCompressionResistancePriority_forOrientation_(NSLayoutPriorityDefaultLow, NSUserInterfaceLayoutOrientationHorizontal)
        except Exception:
            pass

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

        # progress bar (sleeker) inside a full-width container so width never changes
        progressContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        progressContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.infoStack.addArrangedSubview_(progressContainer)
        # Pin this arranged subview to the stack's readable width to prevent overflow
        NSLayoutConstraint.activateConstraints_([
            progressContainer.leadingAnchor().constraintEqualToAnchor_(self.infoStack.leadingAnchor()),
            progressContainer.trailingAnchor().constraintEqualToAnchor_(self.infoStack.trailingAnchor()),
        ])
        progressContainer.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)
        NSLayoutConstraint.activateConstraints_([
            progressContainer.heightAnchor().constraintEqualToConstant_(12.0)
        ])
        self.wordProgress = NSProgressIndicator.alloc().initWithFrame_(NSZeroRect)
        self.wordProgress.setIndeterminate_(False)
        self.wordProgress.setMinValue_(0.0)
        self.wordProgress.setMaxValue_(1.0)
        self.wordProgress.setDoubleValue_(0.0)
        self.wordProgress.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.wordProgress.setWantsLayer_(True)
        if self.wordProgress.layer():
            self.wordProgress.layer().setCornerRadius_(3.0)
        progressContainer.addSubview_(self.wordProgress)
        NSLayoutConstraint.activateConstraints_([
            self.wordProgress.leadingAnchor().constraintEqualToAnchor_(progressContainer.leadingAnchor()),
            self.wordProgress.trailingAnchor().constraintEqualToAnchor_(progressContainer.trailingAnchor()),
            self.wordProgress.centerYAnchor().constraintEqualToAnchor_(progressContainer.centerYAnchor()),
            self.wordProgress.heightAnchor().constraintEqualToConstant_(8.0)
        ])

        # separator line - ensure it respects infoStack width
        separator = NSView.alloc().initWithFrame_(NSZeroRect)
        separator.setTranslatesAutoresizingMaskIntoConstraints_(False)
        separator.setWantsLayer_(True)
        if separator.layer():
            separator.layer().setBackgroundColor_(NSColor.separatorColor().CGColor())
        self.infoStack.addArrangedSubview_(separator)
        NSLayoutConstraint.activateConstraints_([
            separator.leadingAnchor().constraintEqualToAnchor_(self.infoStack.leadingAnchor()),
            separator.trailingAnchor().constraintEqualToAnchor_(self.infoStack.trailingAnchor()),
        ])
        separator.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationHorizontal)
        NSLayoutConstraint.activateConstraints_([
            separator.heightAnchor().constraintEqualToConstant_(1.0)
        ])

        # upcoming words header with better styling
        upcomingLabel = self._label("UPCOMING WORDS", 11.0, NSColor.secondaryLabelColor())
        upcomingLabel.setFont_(NSFont.systemFontOfSize_weight_(11.0, 0.3))  # semibold, uppercase
        self.infoStack.addArrangedSubview_(upcomingLabel)
        # keep label aligned with stack width
        try:
            NSLayoutConstraint.activateConstraints_([
                upcomingLabel.leadingAnchor().constraintEqualToAnchor_(self.infoStack.leadingAnchor()),
                upcomingLabel.trailingAnchor().constraintEqualToAnchor_(self.infoStack.trailingAnchor()),
            ])
        except Exception:
            pass

        # scrollable upcoming words list inside a non-vibrant container to avoid text blending issues
        self.upContainer = NSView.alloc().initWithFrame_(NSZeroRect)
        self.upContainer.setWantsLayer_(True)
        if self.upContainer.layer():
            self.upContainer.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.12, 1.0).CGColor())
        self.upContainer.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.infoStack.addArrangedSubview_(self.upContainer)
        NSLayoutConstraint.activateConstraints_([
            self.upContainer.leadingAnchor().constraintEqualToAnchor_(self.infoStack.leadingAnchor()),
            self.upContainer.trailingAnchor().constraintEqualToAnchor_(self.infoStack.trailingAnchor()),
        ])
        NSLayoutConstraint.activateConstraints_([
            self.upContainer.heightAnchor().constraintGreaterThanOrEqualToConstant_(240.0)
        ])

        # use Apple's factory to get a properly configured scrollable text view
        self.upScroll = NSTextView.scrollableTextView()
        self.upScroll.setHasVerticalScroller_(True)
        self.upScroll.setDrawsBackground_(False)
        self.upScroll.setBorderType_(0)  # no border
        self.upScroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.upContainer.addSubview_(self.upScroll)
        NSLayoutConstraint.activateConstraints_([
            self.upScroll.leadingAnchor().constraintEqualToAnchor_constant_(self.upContainer.leadingAnchor(), 8.0),
            self.upScroll.trailingAnchor().constraintEqualToAnchor_constant_(self.upContainer.trailingAnchor(), -8.0),
            self.upScroll.topAnchor().constraintEqualToAnchor_constant_(self.upContainer.topAnchor(), 8.0),
            self.upScroll.bottomAnchor().constraintEqualToAnchor_constant_(self.upContainer.bottomAnchor(), -8.0)
        ])

        # make scroll view stretch to fill remaining space
        self.upScroll.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultLow - 1, NSUserInterfaceLayoutOrientationVertical)

        # get the embedded text view from the factory scroll view
        self.upcoming = self.upScroll.documentView()
        self.upcoming.setEditable_(False)
        self.upcoming.setSelectable_(True)
        # draw a solid background to avoid vibrancy/clear blending
        self.upcoming.setDrawsBackground_(True)
        self.upcoming.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.12, 1.0))
        # keep rich text enabled so attributed coloring works
        try:
            self.upcoming.setRichText_(True)
        except Exception:
            pass
        try:
            self.upcoming.setFont_(NSFont.userFixedPitchFontOfSize_(13.0))
        except Exception:
            self.upcoming.setFont_(NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0))
        # force high-contrast text color
        self.upcoming.setTextColor_(NSColor.whiteColor())
        # CRITICAL FIX: Keep richText enabled so text rendering works properly
        # We can still use setString_ with rich text enabled
        self.upcoming.setTextContainerInset_(NSMakeSize(8.0, 8.0))
        self.upcoming.setMinSize_(NSMakeSize(0.0, 0.0))
        self.upcoming.setMaxSize_(NSMakeSize(1e7, 1e7))
        self.upcoming.setVerticallyResizable_(True)
        self.upcoming.setHorizontallyResizable_(False)
        # ensure text wraps properly and the container tracks width
        contentSize = self.upScroll.contentSize()
        self.upcoming.textContainer().setWidthTracksTextView_(True)
        self.upcoming.textContainer().setContainerSize_(NSMakeSize(contentSize.width, 1e7))
        self.upcoming.setMinSize_(NSMakeSize(0.0, contentSize.height))
        self.upcoming.setMaxSize_(NSMakeSize(1e7, 1e7))
        try:
            self.upcoming.setAutoresizingMask_(18)  # width + height sizable
        except Exception:
            pass

        # logs view (initially hidden, clean design)
        # factory scrollable text view for logs as well
        self.logsScroll = NSTextView.scrollableTextView()
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

        # get the embedded logs text view
        self.text = self.logsScroll.documentView()
        self.text.setEditable_(False)
        self.text.setSelectable_(True)
        # solid background for reliable contrast
        self.text.setDrawsBackground_(True)
        self.text.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.12, 1.0))
        try:
            self.text.setRichText_(True)
        except Exception:
            pass
        try:
            self.text.setFont_(NSFont.userFixedPitchFontOfSize_(13.0))
        except Exception:
            self.text.setFont_(NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0))
        self.text.setTextColor_(NSColor.whiteColor())
        self.text.setTextContainerInset_(NSMakeSize(12.0, 12.0))
        self.text.setMinSize_(NSMakeSize(0.0, 0.0))
        self.text.setMaxSize_(NSMakeSize(1e7, 1e7))
        self.text.setVerticallyResizable_(True)
        self.text.setHorizontallyResizable_(False)
        # ensure caret/text remain visible in dark appearance
        try:
            self.text.setInsertionPointColor_(NSColor.labelColor())
        except Exception:
            pass
        # configure wrapping and sizing
        logsContentSize = self.logsScroll.contentSize()
        self.text.textContainer().setWidthTracksTextView_(True)
        self.text.textContainer().setContainerSize_(NSMakeSize(logsContentSize.width, 1e7))
        self.text.setMinSize_(NSMakeSize(0.0, logsContentSize.height))
        self.text.setMaxSize_(NSMakeSize(1e7, 1e7))
        try:
            self.text.setAutoresizingMask_(18)
        except Exception:
            pass

        self.window.makeKeyAndOrderFront_(None)
        NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

        self.logger = Logger(self.text)
        self.solver = None
        self.worker = None
        self.isRunning = False
        self.all_words = []

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

    def _setup_app_menu(self):
        """setup application menu with Quit command for Cmd+Q"""
        try:
            from AppKit import NSMenu, NSMenuItem
            
            # create main menu bar
            menubar = NSMenu.alloc().init()
            
            # create app menu (first menu)
            appMenuItem = NSMenuItem.alloc().init()
            menubar.addItem_(appMenuItem)
            
            # create app submenu
            appMenu = NSMenu.alloc().init()
            appMenuItem.setSubmenu_(appMenu)
            
            # add Quit item with Cmd+Q
            quitTitle = "Quit WordHunt Solver"
            quitMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                quitTitle, "terminate:", "q"
            )
            appMenu.addItem_(quitMenuItem)
            
            # set as main menu
            self.app.setMainMenu_(menubar)
        except Exception as e:
            print(f"Menu setup error: {e}")

    def _show_calibration_alert(self) -> bool:
        """present calibration instructions; return True if user proceeds"""
        try:
            from AppKit import NSAlert, NSAlertStyleInformational
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Calibration")
            alert.setInformativeText_((
                "Calibration requires two points:\n"
                "1) Hover over the CENTER of the TOP‑LEFT tile, then wait for the 3‑second countdown.\n"
                "2) Next, hover over the CENTER of the BOTTOM‑RIGHT tile, then wait for the countdown again.\n\n"
                "Tips:\n"
                "• Make sure the full 4×4 board is visible on iPhone Mirroring.\n"
                "• Keep the mouse steady while the countdown runs."
            ))
            try:
                alert.setAlertStyle_(NSAlertStyleInformational)
            except Exception:
                pass
            alert.addButtonWithTitle_("Start Calibration")
            alert.addButtonWithTitle_("Cancel")
            resp = alert.runModal()
            # NSAlertFirstButtonReturn is 1000 typically; treat > 0 as proceed if first button
            return bool(resp == 1000)
        except Exception:
            # if alert fails, proceed in console mode
            print("Calibration instructions: open iPhone Mirroring -> WordHunt board -> keep inputs idle during capture.")
            return True

    def _button(self, title: str, selector_name: str, isPrimary: bool = False):
        """create a properly styled button"""
        btn = NSButton.alloc().initWithFrame_(NSZeroRect)
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setTarget_(self)
        btn.setAction_(selector_name)
        btn.setTranslatesAutoresizingMaskIntoConstraints_(False)
        btn.setContentHuggingPriority_forOrientation_(NSLayoutPriorityDefaultHigh, NSUserInterfaceLayoutOrientationHorizontal)
        
        # improve font
        btn.setFont_(NSFont.systemFontOfSize_weight_(13.0, 0.1))  # slightly bolder
        
        # add height constraint for consistency
        NSLayoutConstraint.activateConstraints_([
            btn.heightAnchor().constraintEqualToConstant_(28.0)
        ])
        
        if isPrimary:
            btn.setKeyEquivalent_("\r")
        
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
            # create symbol image with proper configuration
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
            if img:
                # try to apply configuration if available (macOS 11+)
                try:
                    if hasattr(NSImageSymbolConfiguration, 'configurationWithPointSize_weight_scale_'):
                        cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(14.0, 4, 2)
                        if hasattr(img, 'imageByApplyingSymbolConfiguration_'):
                            img = img.imageByApplyingSymbolConfiguration_(cfg)
                except Exception:
                    pass  # use default size if configuration fails
                
                btn.setImage_(img)
                btn.setImagePosition_(NSImageLeft)
                
                # set content tint color for the icon
                try:
                    btn.setContentTintColor_(color)
                except Exception:
                    pass
                
                # ensure button shows both image and text
                try:
                    btn.setImageHugsTitle_(True)
                except Exception:
                    pass
        except Exception as e:
            # silently fail if SF Symbols not available
            pass

    def _render_upcoming_list(self, current_index: int):
        """render upcoming words with completed ones grayed out and current highlighted"""
        def _apply():
            try:
                from AppKit import NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName
                from Foundation import NSMutableAttributedString
            except Exception:
                NSColor = None
            if not isinstance(self.all_words, list):
                self.all_words = []
            # show up to 200 for performance
            words = self.all_words[:200]
            lines = [f"{idx+1:>3}. {w}" for idx, w in enumerate(words)]
            text = "\n".join(lines)
            if not text:
                self.upcoming.setString_("")
                return
            if NSColor is None:
                self.upcoming.setString_(text)
                return
            base_attrs = {
                NSFontAttributeName: NSFont.monospacedSystemFontOfSize_weight_(13.0, 0.0),
                NSForegroundColorAttributeName: NSColor.whiteColor(),
            }
            mattr = NSMutableAttributedString.alloc().initWithString_attributes_(text, base_attrs)
            total_len = len(text)
            # compute line starts
            starts = []
            pos = 0
            for line in lines:
                starts.append(pos)
                pos += len(line) + 1  # +1 for newline
            # completed range
            if current_index > 0 and len(starts) > 0:
                end = starts[min(current_index, len(starts))] if current_index < len(starts) else total_len
                if end > 0:
                    mattr.addAttribute_value_range_(NSForegroundColorAttributeName, NSColor.tertiaryLabelColor(), NSMakeRange(0, end))
            # set attributed content and force layout/display
            self.upcoming.textStorage().setAttributedString_(mattr)
            try:
                self.upcoming.layoutManager().ensureLayoutForTextContainer_(self.upcoming.textContainer())
                self.upcoming.setNeedsDisplay_(True)
            except Exception:
                pass
            try:
                print(f"upcoming_len={total_len} lines={len(lines)}")
            except Exception:
                pass
        # ensure UI updates happen on the main thread
        try:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(_apply)
        except Exception:
            _apply()

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
        # show guided instructions as an alert/sheet
        if not self._show_calibration_alert():
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
        """update UI for running state with color coding"""
        self.isRunning = running
        self.btnStart.setEnabled_(not running)
        self.btnDry.setEnabled_(not running)
        self.btnCal.setEnabled_(not running)
        self.btnPerm.setEnabled_(not running)
        self.btnMove.setEnabled_(not running)
        self.btnReload.setEnabled_(True)
        self.btnStop.setEnabled_(True)
        
        status_text = "Idle"
        status_color = NSColor.systemGreenColor()  # green for idle
        
        if running:
            status_text = mode if mode else "Running"
            status_color = NSColor.systemOrangeColor()  # orange for running
        
        self.status.setStringValue_(status_text)
        self.status.setTextColor_(status_color)
        
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
                    # resize board container to hug incoming image (within sensible bounds)
                    try:
                        size = nsimg.size()
                        max_side = 420.0
                        min_side = 260.0
                        side = float(max(size.width, size.height))
                        # clamp
                        if side > max_side:
                            side = max_side
                        if side < min_side:
                            side = min_side
                        if hasattr(self, 'boardWidthConstraint') and hasattr(self, 'boardHeightConstraint'):
                            self.boardWidthConstraint.setConstant_(side)
                            self.boardHeightConstraint.setConstant_(side)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error updating board: {e}")
        
        elif event == 'words_list' and 'words' in payload:
            # save and render full list
            self.all_words = [w.upper() for w in payload['words']]
            self._render_upcoming_list(0)
            self.wordProgress.setMinValue_(0)
            self.wordProgress.setMaxValue_(max(1, len(self.all_words)))
            self.wordProgress.setDoubleValue_(0)
            word_label = "word" if len(self.all_words) == 1 else "words"
            self.wordCounter.setStringValue_(f"0 / {len(self.all_words)} {word_label}")
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
            # gray out completed words
            try:
                self._render_upcoming_list(idx)
            except Exception:
                pass

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
