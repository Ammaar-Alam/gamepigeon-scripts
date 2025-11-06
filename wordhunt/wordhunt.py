import mss
import numpy as np
import cv2
import pyautogui
import time
from PIL import Image
# torch, nn, F, transforms moved into WordHuntSolver.__init__
from typing import List, Tuple, Set, Dict, Optional, Callable, Any
import os
import sys
import threading
import tempfile
# import pickle # Removed for reverting dictionary loading
import subprocess
# Avoid importing `keyboard` globally in GUI mode (it can touch HIToolbox/TIS).
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSourceCreate,
    CGEventSourceSetLocalEventsSuppressionInterval,
    CGEventSourceSetLocalEventsFilterDuringSuppressionState,
    kCGEventSourceStateHIDSystemState,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
    kCGHIDEventTap,
    CGPointMake,
    CGEventSetIntegerValueField,
    kCGMouseEventPressure,
    kCGEventMouseMoved,
    kCGEventFilterMaskPermitAllEvents,
    kCGEventSuppressionStateSuppressionInterval
)

# Import additional required Quartz modules
import Quartz

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key: str, store_word: Optional[str] = None):
        """Insert a word by key sequence, but store the original resulting word.

        - key: traversal characters for trie (can be transformed, e.g., 'qu' -> 'q').
        - store_word: original word to return when this node completes.
        """
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = store_word if store_word is not None else key

def create_mouse_event(type, point, source=None):
    """Create a mouse event with proper source state and pressure."""
    if source is None:
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    event = CGEventCreateMouseEvent(source, type, point, kCGMouseButtonLeft)
    # Add pressure for more reliable mouse events
    CGEventSetIntegerValueField(event, kCGMouseEventPressure, 255)
    return event

def find_iphone_mirror_window():
    """Find the iPhone Mirror window and get its position."""
    try:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )

        for window in windows:
            name = window.get(Quartz.kCGWindowName, '')
            owner = window.get(Quartz.kCGWindowOwnerName, '')

            if 'iPhone Mirroring' in str(owner):  # Updated to match exact window name
                bounds = window.get(Quartz.kCGWindowBounds)
                if bounds:
                    print(f"Found iPhone Mirroring window: {bounds}")
                    return (
                        int(bounds['X']),
                        int(bounds['Y']),
                        int(bounds['Width']),
                        int(bounds['Height'])
                    )
    except Exception as e:
        print(f"Error finding window: {e}")

    return None

def move_iphone_mirror_to_corner():
    """Move iPhone Mirroring window to bottom right corner and focus it."""
    script = """
    tell application "System Events"
        tell process "iPhone Mirroring"
            set frontmost to true
            delay 0.5
            click menu item "Bottom Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1
        end tell
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.5)
    except Exception as e:
        print(f"AppleScript failed: {e}")

    window_info = find_iphone_mirror_window()
    if window_info:
        print(f"Window position after move: {window_info}")
        return window_info
    return None

def focus_iphone_window():
    """Focus the iPhone Mirroring window."""
    script = """
    tell application "System Events"
        tell process "iPhone Mirroring"
            set frontmost to true
        end tell
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.1)
    except Exception as e:
            print(f"Focus failed: {e}")

# LetterCNN class definition moved into WordHuntSolver.__init__

def preprocess_for_cnn(cell_img: np.ndarray) -> 'torch.Tensor': # Use string literal for type hint
    """Preprocess image for CNN input."""
    # Convert to grayscale if needed
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img

    # Resize to fixed size
    resized = cv2.resize(gray, (28, 28))

    # Convert to PIL Image
    pil_img = Image.fromarray(resized)

    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Apply transforms
    tensor = transform(pil_img)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    return tensor

def train_model(model, train_loader, device):
    """Training function for the CNN."""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):  # Number of epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

# load_model function removed as model loading is handled in WordHuntSolver.__init__

# (Deduped) create_mouse_event: consolidated above to include pressure.

class WordHuntSolver:
    def __init__(self, no_windows: bool = False, ui_call: Optional[Callable[..., Any]] = None):
        def _resource_path(name: str) -> str:
            """Resolve resource path for both PyInstaller and py2app bundles.

            Order:
            - PyInstaller: sys._MEIPASS/<name>
            - py2app:    Contents/Resources/<name>
            - Fallback:  directory of this file
            """
            # PyInstaller
            if hasattr(sys, '_MEIPASS'):
                p = os.path.join(getattr(sys, '_MEIPASS'), name)
                if os.path.exists(p):
                    return p
            # py2app
            try:
                app_dir = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', 'Resources'))
                p = os.path.join(app_dir, name)
                if os.path.exists(p):
                    return p
            except Exception:
                pass
            # Fallback to source dir
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

        self._resource_path = _resource_path
        # Main-thread dispatcher for macOS UI calls (Quartz/AppKit/pyautogui, etc.)
        # When None, execute directly (CLI mode)
        self._ui_call = ui_call or (lambda f, *a, **k: f(*a, **k))
        # Initialize window position using the accurate window finder
        window_info = self._ui_call(find_iphone_mirror_window)
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")

        # Get screen dimensions for coordinate conversion
        screen_width, screen_height = self._ui_call(pyautogui.size)
        print(f"Screen dimensions: {screen_width}x{screen_height}")

        # Store original window info
        self.window_x, self.window_y, self.window_width, self.window_height = window_info
        print(f"Window position: ({self.window_x}, {self.window_y}) {self.window_width}x{self.window_height}")

        # Initialize screen capture
        self.sct = mss.mss()
        self.monitor = {
            "top": self.window_y,
            "left": self.window_x,
            "width": self.window_width,
            "height": self.window_height,
            "mon": 1
        }

        print(f"Capture monitor: {self.monitor}")

        # UI/Debug control
        self.no_windows = bool(no_windows or os.getenv('WORDHUNT_NO_WINDOWS') == '1')
        # Optional UI event callback (set by GUI)
        self.ui_event: Optional[Callable[[str, dict], None]] = None

        # --- Deferred Imports and Model Definition ---
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as transforms

        class LetterCNN(nn.Module):
            def __init__(self):
                super(LetterCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.fc1 = nn.Linear(64 * 5 * 5, 128)
                self.fc2 = nn.Linear(128, 26)  # 26 letters
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = x.view(-1, 64 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)
        # --- End Deferred Imports ---

        # Initialize CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LetterCNN().to(self.device) # Uses the locally defined LetterCNN
        self.model_lock = threading.RLock()

        # Store torch and transforms for use in other methods
        self.torch = torch
        self.transforms = transforms
        # Debug flags
        self.debug_detect = os.getenv('WORDHUNT_DEBUG_DETECT', '0') == '1'
        # Verbose logging (heavy prints slow down solving). Default: off.
        self.verbose = os.getenv('WORDHUNT_VERBOSE', '0') == '1'

        # Timing constants (original baseline)
        self._SLEEP_CLICK = 0.35
        self._SLEEP_TRANSITION = 0.30
        self._COOLDOWN_START = 0.8
        self._PRE_FOCUS = 0.45
        self._DOWN_DELAY = 0.03
        self._DRAG_DELAY = 0.03
        self._POST_WORD = 0.070

        # Load trained weights (supports external override)
        self._model_path_in_use = None
        self._model_mtime = None
        self.reload_model(initial=True)

        # Letter mapping
        self.idx_to_letter = {i: chr(i + 65).lower() for i in range(26)}

        # Game board dimensions (4x4 grid)
        self.GRID_SIZE = 4
        self.letters = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        # Create debug window (skip in GUI worker threads)
        if not self.no_windows:
            try:
                cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Debug View", self.window_width, self.window_height)
            except Exception as e:
                print(f"Debug window disabled: {e}")
                self.no_windows = True

        # Create event source for mouse control and disable local suppression so Quartz
        # does not coalesce our synthetic drag events.
        self.event_source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        try:
            CGEventSourceSetLocalEventsSuppressionInterval(self.event_source, 0.0)
            CGEventSourceSetLocalEventsFilterDuringSuppressionState(
                self.event_source,
                kCGEventFilterMaskPermitAllEvents,
                kCGEventSuppressionStateSuppressionInterval
            )
        except Exception as e:
            if self.verbose:
                print(f"Warning: suppression tuning failed: {e}")

        # Load dictionary and build trie
        self.dictionary, self.trie = self.load_dictionary()

        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True

        # Initialize window position
        window_info = self._ui_call(find_iphone_mirror_window)
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")

        self.window_x, self.window_y, self.window_width, self.window_height = window_info
        print(f"Window dimensions: {window_info}")

        # Save original window_y for screen capture
        self.window_y_orig = self.window_y

        # Get screen height
        screen_height = self._ui_call(pyautogui.size)[1]

        # Adjust window_y to match Quartz coordinate system
        self.window_y = screen_height - (self.window_y + self.window_height)
        print(f"Adjusted window_y: {self.window_y}")

        # Initialize screen capture with original window_y
        self.sct = mss.mss()
        self.monitor = {
            "top": self.window_y_orig,
            "left": self.window_x,
            "width": self.window_width,
            "height": self.window_height,
            "mon": 1
        }
        print(f"Capture monitor: {self.monitor}")

        # Add a stop flag and running state
        self.stop_flag = False
        self.running = False

        # Initialize keyboard listener only in CLI mode (ui_call is identity) and when allowed.
        # The `keyboard` package can assert HIToolbox when used from worker threads on macOS.
        if ui_call is None and os.getenv('WORDHUNT_ENABLE_HOTKEYS', '0') == '1':
            try:
                import keyboard  # lazy import to avoid GUI-side effects
                keyboard.on_press(self.on_press)
            except Exception as e:
                print(f"Keyboard listener disabled: {e}")

        # Load persisted calibration for this window size if available
        self.use_calibrated_bounds = False
        try:
            self._load_calibration()
        except Exception as e:
            print(f"Calibration load skipped: {e}")

    # --- Helpers for window/event coordinates ---
    def _quartz_from_window(self, x: float, y: float) -> Tuple[float, float]:
        """Convert window-relative top-left coords -> Quartz screen coords.

        NOTE: The solver previously used `self.window_y + offset_y` directly when
        computing points for Quartz. To remain consistent with the existing swipe
        logic, we keep that convention here instead of subtracting from height.
        """
        return (
            self.window_x + x,
            self.window_y + y
        )

    def _click_quartz(self, x: float, y: float):
        """Synthesize a left click at Quartz screen coordinates."""
        def _post():
            try:
                source = self.event_source
                pt = CGPointMake(x, y)
                down = create_mouse_event(kCGEventLeftMouseDown, pt, source)
                CGEventPost(kCGHIDEventTap, down)
                time.sleep(0.02)
                up = create_mouse_event(kCGEventLeftMouseUp, pt, source)
                CGEventPost(kCGHIDEventTap, up)
            except Exception as e:
                print(f"Quartz click failed: {e}")
        self._ui_call(_post)

    def _window_relative_from_screen(self, sx: float, sy: float) -> Tuple[float, float]:
        """Convert absolute screen coords (pyautogui) to window-relative top-left coords.

        Important: mss screenshots and our board bounds use window_y_orig (top-left origin),
        not the Quartz-adjusted self.window_y. So we must subtract window_y_orig here.
        """
        return sx - self.window_x, sy - self.window_y_orig

    # --- Start button detection & click ---
    def detect_start_button(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return window-relative center (x,y) of the green Start button if visible, else None.

        Heuristic:
        - Find large white panel (low S, high V) near the center.
        - Inside its lower portion, find a wide green pill (H in [35,85], S high).
        """
        if img is None or img.size == 0:
            return None

        # Ensure 3 channels (drop alpha if present)
        bgr = img[:, :, :3].copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # White panel mask
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = bgr.shape[:2]
        best = None
        best_score = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 0.05 * w * h:
                continue
            # Prefer centered rectangles
            cx, cy = x + cw/2, y + ch/2
            center_dist = ((cx - w/2) ** 2 + (cy - h/2) ** 2) ** 0.5
            score = area - 2 * center_dist
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        if not best:
            return None

        x, y, cw, ch = best
        panel_roi = bgr[y:y+ch, x:x+cw]
        if panel_roi.size == 0:
            return None

        # Green mask within lower half of panel
        roi_hsv = cv2.cvtColor(panel_roi, cv2.COLOR_BGR2HSV)
        # Relaxed thresholds to robustly capture the Start pill
        lower_green = np.array([35, 50, 70])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(roi_hsv, lower_green, upper_green)

        # Focus on lower band of the panel where the Start pill sits
        gh = green_mask.shape[0]
        lower_start = int(gh * 0.70)  # search lower 30% of the panel
        lower_slice = green_mask[lower_start:, :]
        if lower_slice.size == 0:
            return None

        lower_slice = cv2.morphologyEx(lower_slice, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        g_contours, _ = cv2.findContours(lower_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not g_contours:
            return None

        # Find the widest green blob, likely the Start button
        btn_best = None
        btn_score = 0
        for gc in g_contours:
            gx, gy, gw, gh2 = cv2.boundingRect(gc)
            area = gw * gh2
            aspect = gw / max(gh2, 1)
            if area < 0.003 * cw * ch:
                continue
            if aspect < 1.8:  # pill-like shape should be wide
                continue
            # prefer blobs near bottom center
            cx = gx + gw / 2
            center_dist = abs(cx - (panel_roi.shape[1] / 2))
            score = area - 2 * center_dist
            if score > btn_score:
                btn_score = score
                # Map y from lower_slice back to full panel
                btn_best = (gx, gy + lower_start, gw, gh2)

        if not btn_best:
            return None

        gx, gy, gw, gh2 = btn_best
        center_x = x + gx + gw/2
        center_y = y + gy + gh2/2 + int(0.10 * gh2)  # bias lower within the button

        return int(center_x), int(center_y)

    def press_start_if_present(self) -> bool:
        """Detect the start button and click it. Returns True if clicked.

        Includes an adaptive vertical sweep around the detected center to
        account for coordinate origin mismatches or window scaling.
        """
        try:
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            center = self.detect_start_button(img)
            if center is None:
                # Fallback: if panel is visible, click near bottom-center
                panel = self._find_white_panel_bounds(img)
                if panel is None:
                    return False
                px, py, pw, ph = panel
                approx_x = px + pw / 2
                approx_y = py + ph * 0.93  # closer to panel bottom where Start sits
                center = (int(approx_x), int(approx_y))
                print("Start button heuristic: using panel lower-center fallback (0.93*h)")
            # Convert window-relative point to Quartz and click
            # Optional manual tuning: WORDHUNT_START_Y_OFFSET (pixels)
            try:
                # Default bias tuned to your current mirror layout; can be overridden by env
                manual_bias = float(os.getenv("WORDHUNT_START_Y_OFFSET", "200"))
            except Exception:
                manual_bias = 0.0

            # Vertical sweep deltas to adapt automatically
            sweep = [0, 12, 24, 36, 52, 72, 96, 120, -12, -24]
            for delta in sweep:
                y_bias = manual_bias + delta
                qx, qy = self._quartz_from_window(center[0], center[1] + y_bias)
                print(f"Start: panel_center=({center[0]}, {center[1]}), y_bias={y_bias}, quartz=({qx:.1f}, {qy:.1f})")
                self._click_quartz(qx, qy)
                time.sleep(self._SLEEP_CLICK)
                # Check if panel is gone after the click
                img2 = np.array(self.sct.grab(self.monitor))
                if not self.is_start_panel_visible(img2):
                    time.sleep(self._SLEEP_TRANSITION)
                    return True
            return False
        except Exception as e:
            print(f"Start detection error: {e}")
            return False

    def is_start_panel_visible(self, img: np.ndarray) -> bool:
        """Return True if the white help panel is visible on screen."""
        try:
            return self._find_white_panel_bounds(img) is not None
        except Exception:
            return False

    def on_press(self, key):
        """Handle keyboard events."""
        try:
            if key.char.lower() == 'q':
                print("\nStopping solver...")
                self.stop_flag = True
        except AttributeError:
            pass  # Special key pressed, ignore it

    def _find_white_panel_bounds(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the large white help panel on the start screen; return (x,y,w,h) or None.

        Uses multiple cues to avoid false positives:
        - Area ratio within [0.12, 0.6] of window area
        - Aspect ratio roughly square/landscape (0.8–2.0)
        - Centered horizontally, lower-half vertically
        - Low mean saturation, high mean value inside ROI
        - Green-ish background ring around ROI
        """
        if img is None or img.size == 0:
            return None
        bgr = img[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # White-ish mask for panel
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 70, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        H, W = bgr.shape[:2]

        # Green mask for background ring check
        lower_green_bg = np.array([35, 40, 40])
        upper_green_bg = np.array([95, 255, 255])
        green_mask_full = cv2.inRange(hsv, lower_green_bg, upper_green_bg)

        best = None
        best_score = -1e9
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            area_ratio = area / float(W * H)
            if not (0.12 <= area_ratio <= 0.6):
                continue
            aspect = cw / max(ch, 1)
            if not (0.8 <= aspect <= 2.0):
                continue
            cx, cy = x + cw/2, y + ch/2
            nx, ny = cx / W, cy / H
            # Prefer horizontally centered, vertically lower-half
            center_penalty = abs(nx - 0.5) * 200 + max(0.0, 0.45 - ny) * 400

            # Whiteness inside ROI
            roi_hsv = hsv[y:y+ch, x:x+cw]
            if roi_hsv.size == 0:
                continue
            mean_s = float(roi_hsv[:, :, 1].mean())
            mean_v = float(roi_hsv[:, :, 2].mean())
            if mean_s > 60 or mean_v < 190:
                continue

            # Green background ring around ROI
            pad = 10
            rx0 = max(0, x - pad)
            ry0 = max(0, y - pad)
            rx1 = min(W, x + cw + pad)
            ry1 = min(H, y + ch + pad)
            ring = green_mask_full[ry0:ry1, rx0:rx1].copy()
            # Carve out the ROI center to measure ring only
            ring[(y-ry0):(y-ry0+ch), (x-rx0):(x-rx0+cw)] = 0
            green_ratio = float(ring.mean()) / 255.0
            if green_ratio < 0.05:
                continue

            score = area - center_penalty + green_ratio * 1000
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        return best

    def detect_game_board(self, img: np.ndarray, *, min_scale: float = 0.0, reject_start_panel: bool = False, update_state: bool = True) -> Tuple[dict, np.ndarray]:
        """Detect the game board and calculate accurate board bounds.

        - min_scale: minimum proportion of window width/height the raw board bounds must cover.
        - reject_start_panel: if True, rejects a board that lies fully within the white start panel.
        """
        debug_img = img.copy()

        # Crop to the region where the game board is likely to be
        height, width = img.shape[:2]
        top_crop = int(height * 0.45)  # Adjust based on your game's layout
        bottom_crop = int(height * 0.85)  # Adjust as needed

        cropped = img[top_crop:bottom_crop, :]
        if self.debug_detect and not self.no_windows:
            try:
                cv2.imshow("Cropped Input", cropped)
                cv2.waitKey(1)
            except Exception:
                pass

        # Convert BGR to HSV
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Define green color range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        if self.debug_detect and not self.no_windows:
            try:
                cv2.imshow("Green Mask", green_mask)
                cv2.waitKey(1)
            except Exception:
                pass

        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise Exception("Could not detect game board!")

        # Find the largest contour (assuming it's the board)
        board_contour = max(contours, key=cv2.contourArea)

        # Draw the selected contour
        if self.debug_detect and not self.no_windows:
            try:
                cv2.drawContours(cropped, [board_contour], -1, (0, 255, 0), 2)
                cv2.imshow("Selected Contour", cropped)
                cv2.waitKey(1)
            except Exception:
                pass

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(board_contour)
        print(f"Raw board bounds: x={x}, y={y}, w={w}, h={h}")

        # Reject boards that are too small (e.g., miniature board on start screen)
        if min_scale > 0.0:
            if (w < self.window_width * min_scale) or (h < self.window_height * min_scale):
                raise Exception("Detected board too small relative to window; likely start-screen thumbnail")

        # Adjust coordinates to account for cropping
        y += top_crop

        # Adjust for margins (use ratios scaled by detected board size).
        # Defaults tuned to match your mirrored window; overridable via env.
        ml_ratio = float(os.getenv('WORDHUNT_MARGIN_LEFT_RATIO', '0.22'))
        mr_ratio = float(os.getenv('WORDHUNT_MARGIN_RIGHT_RATIO', '0.22'))
        mt_ratio = float(os.getenv('WORDHUNT_MARGIN_TOP_RATIO', '0.02'))
        mb_ratio = float(os.getenv('WORDHUNT_MARGIN_BOTTOM_RATIO', '0.27'))

        margin_left = int(w * ml_ratio)
        margin_right = int(w * mr_ratio)
        margin_top = int(h * mt_ratio)
        margin_bottom = int(h * mb_ratio)

        board_left = x + margin_left
        board_right = x + w - margin_right
        board_top = y + margin_top
        board_bottom = y + h - margin_bottom

        # Try to refine the grid using tan tile detection for precise centers
        refined = self._refine_grid_from_tiles(img, (board_left, board_top, board_right - board_left, board_bottom - board_top))
        if refined is not None:
            board_left, board_top, board_right, board_bottom, pitch_x, pitch_y = refined
            cell_size_w = pitch_x
            cell_size_h = pitch_y
        else:
            board_width = board_right - board_left
            board_height = board_bottom - board_top
            cell_size_w = board_width / self.GRID_SIZE
            cell_size_h = board_height / self.GRID_SIZE

        # Optionally reject a board within start panel (start screen)
        if reject_start_panel:
            panel = self._find_white_panel_bounds(img)
            if panel is not None:
                px, py, pw, ph = panel
                # If the board rect is fully inside the panel rect, reject
                if (x >= px and y >= py and x + w <= px + pw and y + h <= py + ph):
                    raise Exception("Detected board lies within start panel; rejecting")

        # Build bounds object and optionally persist
        bounds = {
            'top': board_top,
            'bottom': board_bottom,
            'left': board_left,
            'right': board_right,
            'cell_size_w': cell_size_w,
            'cell_size_h': cell_size_h
        }
        if update_state:
            self.board_bounds = bounds

        # Calculate cell positions
        self.cell_positions = {}
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                center_x = board_left + (j + 0.5) * cell_size_w
                center_y = board_top + (i + 0.5) * cell_size_h
                self.cell_positions[(i, j)] = (
                    self.window_x + center_x,
                    self.window_y + center_y
                )

        # Extract board image
        board_img = img[
            int(board_top):int(board_bottom),
            int(board_left):int(board_right)
        ]

        print(f"Board bounds: {bounds}")
        return bounds, board_img

    def _refine_grid_from_tiles(self, full_img: np.ndarray, rect: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int,float,float]]:
        """Refine grid rectangle and pitch using tan tile detection.

        Returns (left, top, right, bottom, pitch_x, pitch_y) in full image coords,
        or None if detection fails.
        """
        x, y, w, h = rect
        roi = full_img[y:y+h, x:x+w]
        if roi is None or roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Tan tile color range (wide, tunable)
        lower_tan = np.array([10, 25, 110])
        upper_tan = np.array([40, 200, 255])
        tan_mask = cv2.inRange(hsv, lower_tan, upper_tan)
        tan_mask = cv2.morphologyEx(tan_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        tan_mask = cv2.morphologyEx(tan_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        # Try 1) projection peaks method (robust to irregular contours)
        proj = self._refine_from_projections(tan_mask)
        if proj is not None:
            left, top, right, bottom, pitch_x, pitch_y = proj
            return x + left, y + top, x + right, y + bottom, pitch_x, pitch_y

        contours, _ = cv2.findContours(tan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter likely tile squares by area and aspect
        roi_area = w * h
        tiles = []
        for c in contours:
            rx, ry, rw, rh = cv2.boundingRect(c)
            area = rw * rh
            if area < 0.004 * roi_area or area > 0.08 * roi_area:
                continue
            aspect = rw / max(rh, 1)
            if 0.7 <= aspect <= 1.3:
                cx = rx + rw / 2.0
                cy = ry + rh / 2.0
                tiles.append((cx, cy, rw, rh))

        if len(tiles) < 12:
            return None

        # Keep the 16 closest to median area if we have too many
        areas = [tw*th for (_,_,tw,th) in tiles]
        median_area = np.median(areas)
        tiles.sort(key=lambda t: abs((t[2]*t[3]) - median_area))
        tiles = tiles[:16] if len(tiles) > 16 else tiles

        # Sort by y to create 4 rows, then by x within rows
        tiles.sort(key=lambda t: t[1])
        rows = [sorted(tiles[i::4], key=lambda t: t[0]) for i in range(4)] if len(tiles) == 16 else []
        if len(rows) != 4 or any(len(r) != 4 for r in rows):
            # Fallback simple chunking
            chunk = max(1, len(tiles)//4)
            rows = [sorted(tiles[i*chunk:(i+1)*chunk], key=lambda t: t[0]) for i in range(4)]
            if any(len(r) == 0 for r in rows):
                return None

        # Compute pitch using median center-to-center distances
        dxs = []
        dys = []
        for r in rows:
            r.sort(key=lambda t: t[0])
            for j in range(3):
                dxs.append(r[j+1][0] - r[j][0])
        for c in range(4):
            col = [rows[i][c] for i in range(len(rows)) if len(rows[i])>c]
            col.sort(key=lambda t: t[1])
            for i in range(3):
                dys.append(col[i+1][1] - col[i][1])
        if not dxs or not dys:
            return None
        pitch_x = float(np.median(dxs))
        pitch_y = float(np.median(dys))

        # Compute grid rectangle from min/max centers ± half pitch
        all_cx = [t[0] for r in rows for t in r]
        all_cy = [t[1] for r in rows for t in r]
        left = int(x + min(all_cx) - pitch_x/2)
        right = int(x + max(all_cx) + pitch_x/2)
        top = int(y + min(all_cy) - pitch_y/2)
        bottom = int(y + max(all_cy) + pitch_y/2)

        # Sanity clamp to original rect
        left = max(x, left)
        right = min(x + w, right)
        top = max(y, top)
        bottom = min(y + h, bottom)

        # Only accept if rectangle is reasonably large and ordered
        if right - left < w * 0.4 or bottom - top < h * 0.4:
            return None
        return left, top, right, bottom, pitch_x, pitch_y

    def _refine_from_projections(self, tan_mask: np.ndarray) -> Optional[Tuple[int,int,int,int,float,float]]:
        """Refine grid using 1D projections of tan pixels to locate 4 columns and 4 rows.

        Returns (left, top, right, bottom, pitch_x, pitch_y) in ROI coordinates.
        """
        H, W = tan_mask.shape[:2]
        if H < 10 or W < 10:
            return None

        col_sum = tan_mask.sum(axis=0).astype(np.float32)
        row_sum = tan_mask.sum(axis=1).astype(np.float32)

        def smooth(a, win):
            win = max(3, int(win))
            k = np.ones(win, dtype=np.float32) / win
            return np.convolve(a, k, mode='same')

        col_s = smooth(col_sum, W/32)
        row_s = smooth(row_sum, H/32)

        def top_k_peaks(arr, k, min_dist):
            idxs = np.argsort(arr)[::-1]  # descending by value
            picked = []
            for idx in idxs:
                if len(picked) >= k:
                    break
                if all(abs(idx - p) >= min_dist for p in picked):
                    picked.append(int(idx))
            picked.sort()
            return picked if len(picked) == k else None

        cols = top_k_peaks(col_s, 4, int(W*0.15))
        rows = top_k_peaks(row_s, 4, int(H*0.15))
        if cols is None or rows is None:
            return None

        cols.sort(); rows.sort()
        dx = np.median(np.diff(cols))
        dy = np.median(np.diff(rows))
        if dx <= 0 or dy <= 0:
            return None

        left = int(cols[0] - dx/2)
        right = int(cols[-1] + dx/2)
        top = int(rows[0] - dy/2)
        bottom = int(rows[-1] + dy/2)

        # Clamp to ROI bounds
        left = max(0, left); top = max(0, top)
        right = min(W, right); bottom = min(H, bottom)
        if right - left < W * 0.4 or bottom - top < H * 0.4:
            return None
        return left, top, right, bottom, float(dx), float(dy)

    def load_dictionary(self) -> Tuple[Set[str], Trie]:
        """Load a dictionary of valid words, merge user whitelist/blacklist, build trie.

        Notes:
        - Word length bounds are configurable via env: `WORDHUNT_MIN_WORD_LEN` (default 3)
          and `WORDHUNT_MAX_WORD_LEN` (default 16). The game board is 4x4 so the longest
          possible word is 16 without tile reuse.
        - Input files accept one word per line. Non A–Z characters are ignored.
        """
        def read_wordlist(path: str) -> Set[str]:
            result: Set[str] = set()
            try:
                with open(path, 'r') as f:
                    for raw in f:
                        line = raw.strip().lower()
                        if not line or line.startswith('#'):
                            continue
                        # Only alphabetic words
                        if line.isalpha():
                            result.add(line)
            except FileNotFoundError:
                return set()
            return result

        # Base dictionary: env override or local/bundled file
        default_dict = self._resource_path("dictionary.txt")
        dict_path = os.path.expanduser(os.getenv("WORDHUNT_DICTIONARY", default_dict))

        print(f"Loading dictionary from {dict_path}...")
        base_words = read_wordlist(dict_path)

        # User filters (env override or ~/.wordhunt)
        default_home = os.path.expanduser("~/.wordhunt")
        whitelist_path = os.path.expanduser(os.getenv("WORDHUNT_WHITELIST", os.path.join(default_home, "whitelist.txt")))
        blacklist_path = os.path.expanduser(os.getenv("WORDHUNT_BLACKLIST", os.path.join(default_home, "blacklist.txt")))

        # Ensure config dir and default files exist to reduce manual setup
        try:
            os.makedirs(os.path.dirname(whitelist_path), exist_ok=True)
            if not os.path.exists(whitelist_path):
                with open(whitelist_path, 'w') as wf:
                    wf.write("# Add words to always allow, one per line.\n")
            if not os.path.exists(blacklist_path):
                with open(blacklist_path, 'w') as bf:
                    bf.write("# Add words to always block, one per line.\n")
        except Exception:
            pass

        whitelist = read_wordlist(whitelist_path)
        blacklist = read_wordlist(blacklist_path)

        if whitelist:
            print(f"Merging whitelist: +{len(whitelist)} words from {whitelist_path}")
        if blacklist:
            print(f"Applying blacklist: -{len(blacklist)} words from {blacklist_path}")

        # Merge filters
        words = (base_words | whitelist) - blacklist
        print(f"Dictionary loaded: base={len(base_words)}, final={len(words)}")

        # Build trie using compressed keys so a 'qu' sequence maps to a single 'q' tile.
        # This matches Word Hunt's QU tile behavior without extra tile steps.
        trie = Trie()
        for w in words:
            key = w.replace('qu', 'q')
            trie.insert(key, store_word=w)
        return words, trie

    @staticmethod
    def _word_score(word: str) -> int:
        """Return Word Hunt point value for a given word length."""
        n = len(word)
        if n <= 2:
            return 0
        if n == 3:
            return 100
        if n == 4:
            return 400
        if n == 5:
            return 800
        if n == 6:
            return 1400
        if n == 7:
            return 1800
        if n == 8:
            return 2200
        if n >= 9:
            return 2600 + (n - 9) * 400
        return 0

    def capture_board(self) -> np.ndarray:
        """Capture the game board using mss, rejecting start-screen thumbnail boards."""
        # Capture the entire window first
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)

        # Show the full capture for debugging
        if self.debug_detect and not self.no_windows:
            try:
                cv2.imshow("Full Capture", img)
                cv2.waitKey(1)
            except Exception:
                pass

        # Use calibration if present; otherwise detect
        if getattr(self, 'use_calibrated_bounds', False) and hasattr(self, 'cal_board_bounds'):
            b = self.cal_board_bounds
            # Ensure downstream uses calibration values
            self.board_bounds = dict(self.cal_board_bounds)
            board_img = img[int(b['top']):int(b['bottom']), int(b['left']):int(b['right'])]
        else:
            # Detect and extract the game board
            _, board_img = self.detect_game_board(img, min_scale=0.33, reject_start_panel=True)

        # Optional debug overlay to verify bounds and grid
        if os.getenv('WORDHUNT_DEBUG_GRID', '0') == '1':
            self._show_grid_overlay(img.copy())

        # Show the cropped board for debugging
        if board_img is not None and board_img.size > 0 and self.debug_detect and not self.no_windows:
            try:
                cv2.imshow("Board Capture", board_img)
                cv2.waitKey(1)
            except Exception:
                pass
        else:
            print("Error: Board capture is empty!")

        return board_img

    def _show_grid_overlay(self, full_img: np.ndarray):
        """Draw board bounds and 4x4 grid overlay for quick visual verification."""
        try:
            b = self.board_bounds
            # Draw outer board
            cv2.rectangle(full_img, (int(b['left']), int(b['top'])), (int(b['right']), int(b['bottom'])), (0,0,255), 2)
            # Draw cell grid
            for i in range(1, self.GRID_SIZE):
                # horizontal lines
                y = int(b['top'] + i * b['cell_size_h'])
                cv2.line(full_img, (int(b['left']), y), (int(b['right']), y), (255,0,0), 1)
                # vertical lines
                x = int(b['left'] + i * b['cell_size_w'])
                cv2.line(full_img, (x, int(b['top'])), (x, int(b['bottom'])), (255,0,0), 1)
            # On GUI mode, avoid cv2 windows from worker threads
            if self.no_windows:
                fd, path = tempfile.mkstemp(prefix='wordhunt_overlay_', suffix='.png')
                os.close(fd)
                cv2.imwrite(path, full_img)
                print(f"Overlay saved to: {path}")
            else:
                try:
                    cv2.imshow('Board Overlay', full_img)
                    cv2.waitKey(1)
                except Exception:
                    fd, path = tempfile.mkstemp(prefix='wordhunt_overlay_', suffix='.png')
                    os.close(fd)
                    cv2.imwrite(path, full_img)
                    print(f"Overlay saved to: {path}")
        except Exception as e:
            print(f"Grid overlay error: {e}")

    # UI event helper
    def _emit(self, event: str, payload: dict):
        cb = getattr(self, 'ui_event', None)
        if cb is None:
            return
        try:
            # Ensure callbacks run on UI/main thread when available
            def _call():
                try:
                    cb(event, payload)
                except Exception:
                    pass
            self._ui_call(_call)
        except Exception:
            pass

    # --- Model override & reloading ---
    def _candidate_model_paths(self) -> list[str]:
        paths = []
        env_path = os.path.expanduser(os.getenv('WORDHUNT_MODEL', '').strip()) if os.getenv('WORDHUNT_MODEL') else ''
        if env_path:
            paths.append(env_path)
        user_path = os.path.expanduser('~/.wordhunt/letter_recognition_model.pth')
        paths.append(user_path)
        paths.append(self._resource_path('letter_recognition_model.pth'))
        # Keep order; first existing will be used
        return paths

    def resolve_model_path(self) -> str:
        for p in self._candidate_model_paths():
            if p and os.path.exists(p):
                return p
        # Fall back to resource path even if missing; caller will raise
        return self._resource_path('letter_recognition_model.pth')

    def reload_model(self, path: Optional[str] = None, initial: bool = False) -> None:
        """Reload CNN weights from override path or bundled model.

        - Respects env WORDHUNT_MODEL, then ~/.wordhunt/letter_recognition_model.pth, then bundled resource.
        - Thread-safe with self.model_lock so it can be called while solving.
        """
        torch = self.torch  # local alias
        candidate = path or self.resolve_model_path()
        if not os.path.exists(candidate):
            msg = f"Model file not found at {candidate}"
            if initial:
                raise FileNotFoundError(msg)
            else:
                print(f"Reload skipped: {msg}")
                return

        try:
            mtime = os.path.getmtime(candidate)
        except Exception:
            mtime = None

        # If same file and unchanged, skip
        if not initial and self._model_path_in_use == candidate and self._model_mtime == mtime:
            print(f"Model unchanged at {candidate}; skipping reload")
            return

        print(f"Attempting to load model from: {candidate}")
        map_location = self.device
        try:
            state = torch.load(candidate, map_location=map_location)
            with self.model_lock:
                self.model.load_state_dict(state)
                self.model.eval()
            self._model_path_in_use = candidate
            self._model_mtime = mtime
            print("Loaded trained model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            if initial:
                raise
            else:
                print("Reload failed; keeping previous model")

    def preprocess_cell(self, cell_img: np.ndarray) -> np.ndarray:
        """Preprocess a cell image for better OCR."""
        # Convert to grayscale if needed
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Initial noise reduction
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        # Binary threshold with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clean up noise and strengthen letter shapes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Add padding (important for Tesseract)
        padding = 20
        padded = cv2.copyMakeBorder(
            binary,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # Scale up image
        scale_factor = 3
        resized = cv2.resize(padded, None,
                            fx=scale_factor,
                            fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        return resized

    def process_image(self, img: np.ndarray) -> List[List[str]]:
        """Process the captured image using trained CNN."""
        if img is None or img.size == 0:
            print("Error: Empty image provided to process_image")
            return [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        letters = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        cell_size_w = self.board_bounds['cell_size_w']
        cell_size_h = self.board_bounds['cell_size_h']
        debug_img = img.copy()

        # Define transforms using the instance variable
        transform = self.transforms.Compose([
            self.transforms.ToPILImage(),
            self.transforms.Grayscale(),
            self.transforms.Resize((28, 28)),
            self.transforms.ToTensor(),
            self.transforms.Normalize((0.5,), (0.5,))
        ])

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                # Extract cell
                margin = 2
                cell_top = max(0, int(i * cell_size_h + margin))
                cell_bottom = min(img.shape[0], int((i + 1) * cell_size_h - margin))
                cell_left = max(0, int(j * cell_size_w + margin))
                cell_right = min(img.shape[1], int((j + 1) * cell_size_w - margin))

                cell_img = img[cell_top:cell_bottom, cell_left:cell_right]

                if cell_img.size == 0 or cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
                    continue

                # Preprocess and get prediction
                try:
                    # Preprocess cell image
                    tensor = transform(cell_img)
                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                    tensor = tensor.to(self.device)

                    # Get prediction
                    with self.model_lock:
                        with self.torch.no_grad(): # Use self.torch
                            output = self.model(tensor)
                        pred_idx = output.argmax(dim=1).item()
                        confidence = self.torch.exp(output.max()).item() # Use self.torch

                        letter = self.idx_to_letter[pred_idx]

                        if confidence > 0.5:  # Confidence threshold
                            letters[i][j] = letter
                            if self.verbose:
                                print(f"Detected letter at {i},{j}: {letter.upper()} (confidence: {confidence:.2f})")
                        else:
                            if self.verbose:
                                print(f"Low confidence detection at {i},{j}: {letter.upper()} ({confidence:.2f})")

                except Exception as e:
                    print(f"Error processing cell {i},{j}: {e}")
                    continue

                # Debug visualization
                cv2.rectangle(
                    debug_img,
                    (cell_left, cell_top),
                    (cell_right, cell_bottom),
                    (0, 255, 0),
                    2
                )
                if letters[i][j]:
                    cv2.putText(
                        debug_img,
                        letters[i][j].upper(),
                        (cell_left + int(cell_size_w * 0.25), cell_top + int(cell_size_h * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

        # Show debug view
        if not self.no_windows:
            try:
                cv2.imshow("Debug View", debug_img)
                cv2.waitKey(1)
            except Exception:
                pass

        return letters

    def find_words(self) -> List[str]:
        """Find all valid words on the board using optimized DFS with trie."""
        def is_valid_pos(x: int, y: int) -> bool:
            return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

        def dfs(x: int, y: int, visited: set, node: TrieNode, words: set):
            if node.is_word:
                words.add(node.word)

            # Check all 8 directions
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (is_valid_pos(new_x, new_y) and
                    (new_x, new_y) not in visited):
                    letter = self.letters[new_x][new_y]
                    if letter in node.children:
                        visited.add((new_x, new_y))
                        dfs(new_x, new_y, visited, node.children[letter], words)
                        visited.remove((new_x, new_y))

        found_words = set()
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.letters[i][j]:  # Only start from cells with letters
                    letter = self.letters[i][j]
                    if letter in self.trie.root.children:
                        visited = {(i, j)}
                        dfs(i, j, visited, self.trie.root.children[letter], found_words)

        return sorted(list(found_words), key=lambda x: (-len(x), x))

    def find_word_path(self, word: str) -> List[Tuple[int, int]]:
        """Find the path of coordinates that spell out the given word."""
        def dfs(x: int, y: int, visited: set, target: str, index: int, path: list) -> bool:
            """Helper function to find path using DFS."""
            if index == len(target):
                return True
        # In this system:
            # - Moving up means decreasing x (going to earlier rows)
            # - Moving down means increasing x (going to later rows)
            # - Moving left means decreasing y
            # - Moving right means increasing y
            directions = [
                (-1,-1),  # up-left
                (-1,0),   # up
                (-1,1),   # up-right
                (0,-1),   # left
                (0,1),    # right
                (1,-1),   # down-left
                (1,0),    # down
                (1,1)     # down-right
            ]

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.GRID_SIZE and
                    0 <= new_y < self.GRID_SIZE and
                    (new_x, new_y) not in visited):
                    tile = self.letters[new_x][new_y].lower()

                    # Handle QU tile: one tile may satisfy two letters 'qu'
                    step = None
                    if index < len(target):
                        if tile == 'q' and index + 1 < len(target) and target[index:index+2] == 'qu':
                            step = 2
                        elif tile == target[index].lower():
                            # Regular one-letter advance (including 'q' not followed by 'u')
                            step = 1

                    if step is not None:
                        visited.add((new_x, new_y))
                        path.append((new_x, new_y))
                        if dfs(new_x, new_y, visited, target, index + step, path):
                            return True
                        path.pop()
                        visited.remove((new_x, new_y))
            return False

        word = word.lower()
        # Search for starting letter
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                tile = self.letters[i][j].lower()
                # Starting tile may be 'q' matching 'qu'
                start_index = None
                if tile == 'q' and len(word) >= 2 and word.startswith('qu'):
                    start_index = 2
                elif tile == word[0]:
                    start_index = 1
                if start_index is not None:
                    path = [(i, j)]
                    visited = {(i, j)}
                    if dfs(i, j, visited, word, start_index, path):
                        # Reverse the vertical coordinates to match game board layout
                        return [(self.GRID_SIZE - 1 - x, y) for x, y in path]
        return []


    def swipe_word(self, word: str):
        """Execute swipe motion to input a word using Quartz events."""
        path = self.find_word_path(word)
        if not path:
            print(f"Could not find path for word: {word}")
            return

        try:
            screen_height = self._ui_call(pyautogui.size)[1]
            points = []

            # Calculate board area within window (heuristic mapping used previously)
            board_height = self.board_bounds['bottom'] - self.board_bounds['top']

            # Move the board down by adjusting the starting position
            board_top = self.window_y + int(self.window_height * 0.25)  # Adjust vertical position
            board_space = self.window_height * 0.25  # Vertical spacing between cells
            board_width = self.window_width * 0.6  # Board takes up 60% of window width
            board_x_offset = self.window_width * 0.2  # 20% margin on each side

            for grid_pos in path:
                i, j = grid_pos

                # Calculate normalized position (0-1 range)
                norm_x = (j + 0.5) / self.GRID_SIZE
                norm_y = (i + 0.5) / self.GRID_SIZE

                # Calculate actual screen coordinates
                screen_x = self.window_x + board_x_offset + (norm_x * board_width)
                y_pos = board_top + (norm_y * board_space)

                # Convert to Quartz coordinate system
                quartz_y = screen_height - y_pos

                if self.verbose:
                    print(f"Grid pos {i},{j} -> Window pos ({screen_x:.1f}, {y_pos:.1f}) -> Quartz pos ({screen_x:.1f}, {quartz_y:.1f})")
                points.append((screen_x, quartz_y))

            # Use direct points (original behavior)

            def _do_swipe():
                # Create event source
                source = self.event_source

                # Mouse down at start
                start_point = CGPointMake(points[0][0], points[0][1])
                down_event = create_mouse_event(kCGEventLeftMouseDown, start_point, source)
                CGEventPost(kCGHIDEventTap, down_event)
                time.sleep(self._DOWN_DELAY)

                # Drag through points
                for point in points[1:]:
                    quartz_point = CGPointMake(point[0], point[1])
                    drag_event = create_mouse_event(kCGEventLeftMouseDragged, quartz_point, source)
                    CGEventPost(kCGHIDEventTap, drag_event)
                    time.sleep(self._DRAG_DELAY)

                # Release at end
                end_point = CGPointMake(points[-1][0], points[-1][1])
                up_event = create_mouse_event(kCGEventLeftMouseUp, end_point, source)
                CGEventPost(kCGHIDEventTap, up_event)
            self._ui_call(_do_swipe)

            # Adaptive delay between words based on swipe complexity
            time.sleep(self._POST_WORD)

        except Exception as e:
            print(f"Error swiping word {word}: {e}")

    def wait_for_round_start(self, max_wait: float = 15.0, require_min_letters: int = 8) -> np.ndarray:
        """Wait until the actual 4x4 board is visible by verifying letter detections.

        Returns the board image when ready, or raises if not found within timeout.
        """
        start = time.time()
        last_error = None
        clicks = 0
        last_click_ts = 0.0
        cooldown = self._COOLDOWN_START
        while time.time() - start < max_wait:
            if self.stop_flag:
                raise TimeoutError("Stopped")
            try:
                self._ui_call(focus_iphone_window)
                screenshot = self.sct.grab(self.monitor)
                img = np.array(screenshot)

                # If the start panel is visible, click Start, then wait a moment to re-check
                if self.is_start_panel_visible(img):
                    now = time.time()
                    if clicks < 4 and (now - last_click_ts) > cooldown:
                        if self.press_start_if_present():
                            clicks += 1
                            last_click_ts = now
                            time.sleep(0.35)
                            # After clicking, let it transition; do not spam clicks
                            continue
                    last_error = "Start panel visible; waiting for it to disappear..."
                    time.sleep(0.25)
                    continue

                # No start panel; use calibration if present, otherwise detect
                if getattr(self, 'use_calibrated_bounds', False) and hasattr(self, 'cal_board_bounds'):
                    # Ensure solver uses calibration bounds
                    self.board_bounds = dict(self.cal_board_bounds)
                    board_img = self.capture_board()
                else:
                    # Detect and set bounds for this attempt
                    bounds, board_img = self.detect_game_board(img, min_scale=0.3, reject_start_panel=False, update_state=True)
                    self.board_bounds = bounds

                letters = self.process_image(board_img)
                non_empty = sum(1 for row in letters for c in row if c)
                if non_empty >= require_min_letters:
                    # Lock to calibration when available
                    if getattr(self, 'use_calibrated_bounds', False) and hasattr(self, 'cal_board_bounds'):
                        self.board_bounds = dict(self.cal_board_bounds)
                    return board_img
                last_error = f"Only {non_empty} cells recognized; waiting for board..."
            except Exception as e:
                last_error = str(e)
            time.sleep(0.25)
        raise TimeoutError(last_error or "Timed out waiting for board")

    def solve(self):
        """Main solving loop."""
        # Reset stop flag for a fresh run
        self.stop_flag = False
        self.running = True
        # Start timer for 1:15 round limit
        start_time = time.time()
        round_duration = 82.5  # 1:15 + 2.5 seconds buffer to be safe

        # Focus window before capture
        self._ui_call(focus_iphone_window)
        time.sleep(self._PRE_FOCUS)

        # Wait for round start if requested (default auto)
        if getattr(self, 'manual_start', False):
            print("\nTap Start on the phone, then press Enter to begin...")
            input()
            print("Capturing board...")
            board_img = self.capture_board()
        else:
            print("\nWaiting for the board to appear...")
            try:
                board_img = self.wait_for_round_start()
                print("Board detected. Proceeding...")
                # Notify UI with board image
                try:
                    self._emit('board', {'image': board_img})
                except Exception:
                    pass
            except Exception as e:
                print(f"Auto-wait failed: {e}. You can retry with --manual-start.")
                return

        print("\nProcessing image...")
        self.letters = self.process_image(board_img)

        print("\nDetected letters:")
        for row in self.letters:
            print(" ".join(letter.upper() if letter else '_' for letter in row))

        # Verify letters are correct (no need for this anymore since AI is hella accurate)
        # print("\nAre these letters correct? (y/n)")
        # if input().lower() != 'y':
        #     print("Aborting...")
        #     return

        # Find all possible words
        words = self.find_words()
        potential_score = sum(self._word_score(w) for w in words)
        print(f"\nFound {len(words)} possible words")
        print(f"Potential max score (all valid): {potential_score}")
        # Notify UI with words list
        try:
            self._emit('words_list', {'words': words})
            self._emit('potential_score', {'score': potential_score})
        except Exception:
            pass


        # Sort words by length (longest first) and alphabetically
        words.sort(key=lambda x: (-len(x), x))

        # Make sure window is still focused before swiping
        self._ui_call(focus_iphone_window)
        time.sleep(self._PRE_FOCUS)  # Wait for focus

        # Execute swipes for each word
        # time.sleep(0.5) # so top word doesn't get skipped // never that didn't fix it :)
        for idx, word in enumerate(words, 1):
            # Check if time limit reached or stop flag set
            elapsed_time = time.time() - start_time
            if elapsed_time >= round_duration:
                print("\nTime limit reached (1:15)! Stopping solver...")
                break

            if self.stop_flag:
                print("Solver stopped by user (Q pressed)")
                break

            print(f"Swiping word: {word}")
            try:
                self._emit('progress', {'current': word, 'index': idx, 'total': len(words)})
            except Exception:
                pass
            # Verify path exists before attempting swipe
            path = self.find_word_path(word)
            if not path:
                print(f"Could not find path for word: {word}")
                continue

            # Add small delay before first swipe
            if word == words[0]:  # If this is the first word
                time.sleep(0.5)  # Add small settle before first swipe
                self._ui_call(focus_iphone_window)  # Re-focus window before first swipe

            if getattr(self, 'no_swipe', False):
                print(f"[dry-run] Would swipe: {word}")
            else:
                if self.stop_flag:
                    break
                self.swipe_word(word)

        self.running = False

    def __del__(self):
        # Stop the keyboard listener
        if hasattr(self, 'listener'):
            self.listener.stop()
        cv2.destroyAllWindows()

    def stop(self):
        """Request stop and try to release mouse if dragging."""
        self.stop_flag = True
        def _release():
            try:
                # Synthesize a mouse up at current position to ensure drag is released
                from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGEventLeftMouseUp, kCGHIDEventTap, CGPointMake, CGEventSourceCreate, kCGEventSourceStateHIDSystemState, kCGMouseButtonLeft
                import pyautogui as pag
                x, y = pag.position()
                src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
                pt = CGPointMake(x, y)
                up = CGEventCreateMouseEvent(src, kCGEventLeftMouseUp, pt, kCGMouseButtonLeft)
                CGEventPost(kCGHIDEventTap, up)
            except Exception:
                pass
        # Try to release on main thread
        self._ui_call(_release)

    def test_bounds(self):
        """Move mouse around the border of the game area to visualize swipe boundaries."""
        try:
            screen_height = self._ui_call(pyautogui.size)[1]

            # Calculate board area
            board_top = self.window_y + int(self.window_height * 0.25) # top margin
            board_space = self.window_height * 0.25 # vertical spacing between cells
            board_width = self.window_width * 0.6 # the width (horizontal spacing) each grid cell
            board_x_offset = self.window_width * 0.2 # where the left edge of the grid starts

            # Calculate corner points (clockwise from top-left)
            corners = [
                # Top-left (0,0)
                (
                    self.window_x + board_x_offset,
                    screen_height - (board_top)
                ),
                # Top-right (0,3)
                (
                    self.window_x + board_x_offset + board_width,
                    screen_height - (board_top)
                ),
                # Bottom-right (3,3)
                (
                    self.window_x + board_x_offset + board_width,
                    screen_height - (board_top + board_space)
                ),
                # Bottom-left (3,0)
                (
                    self.window_x + board_x_offset,
                    screen_height - (board_top + board_space)
                )
            ]

            print("\nMoving mouse to show game boundaries...")
            print("Press Ctrl+C to stop")

            # Move to each corner and pause
            while True:  # Loop until interrupted
                for i, (x, y) in enumerate(corners):
                    def _move():
                        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
                        point = CGPointMake(x, y)
                        move_event = CGEventCreateMouseEvent(source, kCGEventMouseMoved, point, kCGMouseButtonLeft)
                        CGEventPost(kCGHIDEventTap, move_event)
                    self._ui_call(_move)
                    print(f"Corner {i+1}: ({x:.1f}, {y:.1f})")
                    time.sleep(1)  # Pause at each corner

        except KeyboardInterrupt:
            print("\nBoundary test stopped")
        except Exception as e:
            print(f"Error during boundary test: {e}")

    def test_cell_positions(self):
        """Move mouse to center of each cell to visualize all possible swipe positions."""
        try:
            screen_height = self._ui_call(pyautogui.size)[1]

            # Calculate board area
            board_top = self.window_y + int(self.window_height * 0.2)
            board_space = self.window_height * 0.35
            board_width = self.window_width * 0.8
            board_x_offset = self.window_width * 0.1

            print("\nMoving mouse to show all cell positions...")
            print("Press Ctrl+C to stop")

            while True:  # Loop until interrupted
                # Loop through each cell in grid
                for i in range(self.GRID_SIZE):
                    for j in range(self.GRID_SIZE):
                        # Calculate normalized position (0-1 range)
                        norm_x = (j + 0.5) / self.GRID_SIZE
                        norm_y = (i + 0.5) / self.GRID_SIZE

                        # Calculate actual screen coordinates
                        screen_x = self.window_x + board_x_offset + (norm_x * board_width)
                        y_pos = board_top + (norm_y * board_space)
                        quartz_y = screen_height - y_pos

                        # Move to position
                        def _move_cell():
                            point = CGPointMake(screen_x, quartz_y)
                            source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
                            move_event = CGEventCreateMouseEvent(source, kCGEventMouseMoved, point, kCGMouseButtonLeft)
                            CGEventPost(kCGHIDEventTap, move_event)
                        self._ui_call(_move_cell)

                        print(f"Cell ({i},{j}): ({screen_x:.1f}, {quartz_y:.1f})")
                        time.sleep(0.5)  # Pause at each cell

                print("\nCompleted one cycle, starting over...")
                time.sleep(1)  # Pause between cycles

        except KeyboardInterrupt:
            print("\nCell position test stopped")
        except Exception as e:
            print(f"Error during cell position test: {e}")
            import traceback
            traceback.print_exc()

    # --- Calibration persistence ---
    def _config_path(self) -> str:
        return os.path.expanduser("~/.wordhunt/config.json")

    def _load_calibration(self):
        path = self._config_path()
        if not os.path.exists(path):
            return
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        key = f"{self.window_width}x{self.window_height}"
        entry = data.get(key)
        if not entry:
            return
        self.cal_board_bounds = entry['board_bounds']
        self.use_calibrated_bounds = True
        print(f"Loaded calibration for window {key}: {self.cal_board_bounds}")

    def _save_calibration(self):
        os.makedirs(os.path.dirname(self._config_path()), exist_ok=True)
        import json
        data = {}
        if os.path.exists(self._config_path()):
            try:
                with open(self._config_path(), 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        key = f"{self.window_width}x{self.window_height}"
        # Persist the calibration bounds (use current cal bounds if set, else current board)
        to_save = getattr(self, 'cal_board_bounds', getattr(self, 'board_bounds', None))
        if not to_save:
            return
        data[key] = {
            'board_bounds': to_save
        }
        with open(self._config_path(), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration for window {key} -> {self._config_path()}")

    def calibrate(self, delay_seconds: float = 3.0, show_overlay: bool = False):
        """Interactive-but-nonblocking calibration suitable for GUI.

        Workflow:
        - Ask user to hover over top-left tile center; countdown; capture mouse pos
        - Ask user to hover over bottom-right tile center; countdown; capture mouse pos
        - Compute grid, persist, show overlay
        """
        def countdown(label: str, secs: float):
            steps = int(max(1, secs))
            for i in range(steps, 0, -1):
                print(f"{label} in {i}…")
                time.sleep(1)
            # small settle
            time.sleep(max(0.0, secs - steps))

        print("\nCalibration: hover your mouse over the CENTER of the top-left tile.")
        self._ui_call(focus_iphone_window)
        countdown("Capturing top-left", delay_seconds)
        x1, y1 = self._ui_call(pyautogui.position)
        print(f"Top-left center at screen coords: ({x1}, {y1})")

        print("Now hover your mouse over the CENTER of the bottom-right tile.")
        self._ui_call(focus_iphone_window)
        countdown("Capturing bottom-right", delay_seconds)
        x2, y2 = self._ui_call(pyautogui.position)
        print(f"Bottom-right center at screen coords: ({x2}, {y2})")

        # Convert to window-relative coordinates (top-left origin)
        rx1, ry1 = self._window_relative_from_screen(x1, y1)
        rx2, ry2 = self._window_relative_from_screen(x2, y2)
        pitch_x = (rx2 - rx1) / 3.0
        pitch_y = (ry2 - ry1) / 3.0
        left = rx1 - pitch_x / 2.0
        right = rx2 + pitch_x / 2.0
        top = ry1 - pitch_y / 2.0
        bottom = ry2 + pitch_y / 2.0

        self.board_bounds = {
            'top': float(top),
            'bottom': float(bottom),
            'left': float(left),
            'right': float(right),
            'cell_size_w': float(pitch_x),
            'cell_size_h': float(pitch_y)
        }
        # Store separately for future runs and to avoid being overwritten by detection
        self.cal_board_bounds = dict(self.board_bounds)
        self.use_calibrated_bounds = True
        self._save_calibration()
        print("Calibration complete. A debug overlay window will show the grid.")
        if show_overlay:
            try:
                screenshot = self.sct.grab(self.monitor)
                self._show_grid_overlay(np.array(screenshot))
            except Exception as e:
                print(f"Overlay skipped: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Word Hunt Solver")
    parser.add_argument("command", nargs="?", choices=["solve", "dry-run", "menu", "calibrate"], default="menu",
                        help="Action to perform: solve (non-interactive), dry-run (no swipes), or menu (interactive)")
    parser.add_argument("--no-position", action="store_true", dest="no_position",
                        help="Do not auto-move the iPhone Mirroring window")
    parser.add_argument("--manual-start", action="store_true", dest="manual_start",
                        help="Ask user to press Enter after tapping Start on the phone")
    parser.add_argument("--max-wait", type=float, default=15.0,
                        help="Max seconds to wait for board to appear in auto mode")
    args = parser.parse_args()

    try:
        if args.command == "menu":
            print("\nWord Hunt Solver")
            print("1. Open the Word Hunt game")
            print("2. Press Enter to position the window")
            print("3. Options:")
            print("   T - Test boundaries")
            print("   C - Test cell positions")
            print("   Enter - Start solving")
            print("   Q - Quit")
            input()

        # Move window to corner unless suppressed
        if not args.no_position:
            print("\nMoving iPhone window to corner...")
            window_info = move_iphone_mirror_to_corner()
            if not window_info:
                print("Failed to position window!")
                if args.command != "menu":
                    return
        time.sleep(1)

        solver = WordHuntSolver()
        # Configure behavior from args
        solver.manual_start = bool(args.manual_start)
        solver.max_wait = float(args.max_wait)

        # Non-interactive modes
        if args.command == "calibrate":
            solver.manual_start = True
            # Guide into board screen, then calibrate
            print("\nMove to board screen (press Start if needed). Then we'll capture two points.")
            input("Press Enter when the board is visible...")
            solver.calibrate()
            return
        if args.command in ("solve", "dry-run"):
            solver.no_swipe = (args.command == "dry-run")
            solver.solve()
            return

        # Interactive menu
        while True:
            choice = input("\nEnter choice (T/C/Enter/Q): ").lower()
            if choice == 't':
                solver.test_bounds()
            elif choice == 'c':
                solver.test_cell_positions()
            elif choice == 'q':
                break
            elif choice == '':
                solver.solve()
            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\nSolver stopped by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
