import mss.tools
import numpy as np
import pyautogui
import time
import sys
import signal
import cv2
import Quartz
import AppKit
import subprocess
import os
from pynput.mouse import Button, Controller

# Configure pyautogui
pyautogui.FAILSAFE = False
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0
pyautogui.DEFAULT_DURATION = 0

mouse = Controller()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nExiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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
            
            if 'iPhone' in str(name) or 'iPhone' in str(owner):
                bounds = window.get(Quartz.kCGWindowBounds)
                if bounds:
                    print(f"Found iPhone Mirror window: {bounds}")
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
        set mirrorProcess to first process whose name contains "iPhone"
        set frontmost of mirrorProcess to true
        delay 0.5
        click menu item "Bottom Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1 of mirrorProcess
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
        set mirrorProcess to first process whose name contains "iPhone"
        set frontmost of mirrorProcess to true
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.1)
    except Exception as e:
        print(f"Focus failed: {e}")

def hold_mouse_down():
    """Hold down the left mouse button using Quartz."""
    try:
        event = Quartz.CGEventCreateMouseEvent(
            None, 
            Quartz.kCGEventLeftMouseDown, 
            Quartz.CGPointMake(mouse.position[0], mouse.position[1]),
            Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        print("Mouse button held down with Quartz")
    except Exception as e:
        print(f"Error holding mouse down: {e}")

def release_mouse():
    """Release the left mouse button using Quartz."""
    try:
        event = Quartz.CGEventCreateMouseEvent(
            None, 
            Quartz.kCGEventLeftMouseUp, 
            Quartz.CGPointMake(mouse.position[0], mouse.position[1]),
            Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        print("Mouse button released with Quartz")
    except Exception as e:
        print(f"Error releasing mouse: {e}")

class InstagramGame:
    def __init__(self):
        window_info = find_iphone_mirror_window()
        if not window_info:
            print("Could not find iPhone Mirror window!")
            sys.exit(1)
        
        self.window_x, self.window_y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT = window_info
        print(f"Using window dimensions: x={self.window_x}, y={self.window_y}, w={self.WINDOW_WIDTH}, h={self.WINDOW_HEIGHT}")
        
        self.scale_factor = 2
        print(f"Using scale factor: {self.scale_factor}")
        
        self.bar_y = int(self.WINDOW_HEIGHT * 0.93)
        self.play_again_y = int(self.WINDOW_HEIGHT * 0.92)
        
        self.GAME_TOP = 150
        self.GAME_BOTTOM = self.WINDOW_HEIGHT - 50
        print(f"Game area: y={self.GAME_TOP} to y={self.GAME_BOTTOM}")
        
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.velocity_x = 0
        self.velocity_y = 0
        self.stuck_counter = 0
        self.direction = 0
        self.prediction_offset = 50
        self.last_prediction = None
        self.last_score_y = None
        self.score = 0
        
        self.running = False
        self.game_started = False
        self.first_start = True
        self.game_start_time = None
        
        self.sct = mss.mss()
        self.monitor = {
            "top": self.window_y,
            "left": self.window_x,
            "width": self.WINDOW_WIDTH,
            "height": self.WINDOW_HEIGHT,
            "mon": 1
        }
        print(f"Capture area: {self.monitor}")
        
        self.prev_img = None
        self.last_hit_time = None
        self.min_hit_interval = 0.5
    
    def scale_coordinates(self, x, y):
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def update_tracking(self, x, y):
        current_time = time.time()
        
        if self.last_x is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                dx = x - self.last_x
                dy = y - self.last_y
                new_velocity_x = dx / dt
                new_velocity_y = dy / dt
                
                SMOOTHING = 0.7
                self.velocity_x = SMOOTHING * new_velocity_x + (1 - SMOOTHING) * self.velocity_x
                self.velocity_y = SMOOTHING * new_velocity_y + (1 - SMOOTHING) * self.velocity_y
                
                if abs(dx) > 2:
                    self.direction = 1 if dx > 0 else -1
                
                if abs(dx) < 2 and abs(dy) < 2:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                
                if self.last_y and y < self.last_y:
                    if self.last_y > self.GAME_BOTTOM - 100:
                        current_time = time.time()
                        if (self.last_hit_time is None or 
                            current_time - self.last_hit_time > self.min_hit_interval):
                            self.score += 1
                            print(f"Score updated: {self.score}")
                            self.last_hit_time = current_time
        
        self.last_x = x
        self.last_y = y
        self.last_time = current_time
    
    def predict_position(self, x, y):
        if self.velocity_x == 0 and self.last_prediction is not None:
            return self.last_prediction
            
        prediction_time = 0.1
        predicted_x = x + int(self.velocity_x * prediction_time)
        
        if self.direction != 0:
            vertical_factor = min((y - self.GAME_TOP) / (self.GAME_BOTTOM - self.GAME_TOP), 1.0)
            speed_factor = min(abs(self.velocity_x) / 1000, 1.0)
            offset = int(self.prediction_offset * vertical_factor * speed_factor) * self.direction
            predicted_x += offset
        
        predicted_x = max(20, min(predicted_x, self.WINDOW_WIDTH - 20))
        
        self.last_prediction = predicted_x
        return predicted_x

    def detect_emoji(self, screenshot):
        """Enhanced emoji detection using multiple color spaces."""
        try:
            img = np.array(screenshot)
            
            # Convert to multiple color spaces
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # HSV color ranges for emoji
            hsv_ranges = [
                (np.array([20, 100, 100]), np.array([30, 255, 255])),  # Yellow
                (np.array([0, 100, 100]), np.array([10, 255, 255])),   # Red
                (np.array([170, 100, 100]), np.array([180, 255, 255])) # Red (wrap)
            ]
            
            # Combine masks from different color spaces
            combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
            for lower_hsv, upper_hsv in hsv_ranges:
                hsv_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
                combined_mask = cv2.bitwise_or(combined_mask, hsv_mask)
            
            # Add LAB color space detection for yellow
            lab_lower = np.array([150, 120, 150])
            lab_upper = np.array([250, 145, 255])
            lab_mask = cv2.inRange(lab_img, lab_lower, lab_upper)
            combined_mask = cv2.bitwise_or(combined_mask, lab_mask)
            
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area and circularity
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area threshold
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.7:  # Circle has circularity of 1
                                valid_contours.append(contour)
                
                if valid_contours:
                    # Use the largest valid contour
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    x_center = x + w // 2
                    y_center = y + h // 2
                    
                    x_scaled, y_scaled = self.scale_coordinates(x_center, y_center)
                    self.update_tracking(x_scaled, y_scaled)
                    predicted_x = self.predict_position(x_scaled, y_scaled)
                    
                    return predicted_x
            
            return None
            
        except Exception as e:
            print(f"Error in detect_emoji: {e}")
            return None

    def click_play_again(self):
        try:
            screen_x = self.window_x + self.WINDOW_WIDTH//2
            screen_y = self.window_y + self.play_again_y
            
            screen_x = max(0, min(screen_x, self.sct.monitors[1]["width"]))
            screen_y = max(0, min(screen_y, self.sct.monitors[1]["height"]))
            
            pyautogui.moveTo(screen_x, screen_y, duration=0, _pause=False)
            pyautogui.click()
            
            self.game_started = True
            self.first_start = False
            self.game_start_time = time.time()
            
            self.last_x = None
            self.last_y = None
            self.last_time = None
            self.velocity_x = 0
            self.velocity_y = 0
            self.stuck_counter = 0
            self.direction = 0
            self.last_prediction = None
            self.prev_img = None
            self.score = 0
            self.last_hit_time = None
            
        except Exception as e:
            print(f"Error clicking play: {e}")

    def move_bar(self, target_x):
        try:
            screen_x = self.window_x + target_x
            screen_y = self.window_y + self.bar_y

            print(f"Moving bar to screen coordinates: ({screen_x}, {screen_y})")

            screen_x = max(0, min(screen_x, self.sct.monitors[1]["width"] - 1))
            screen_y = max(0, min(screen_y, self.sct.monitors[1]["height"] - 1))

            pyautogui.moveTo(screen_x, screen_y, duration=0, _pause=False)
        except Exception as e:
            print(f"Error moving bar: {e}")

if __name__ == "__main__":
    print("\nINSTRUCTIONS:")
    print("1. Make sure iPhone Mirroring is running")
    print("2. The Instagram game should be ready")
    print("3. Press Enter to start - window will be moved automatically")
    print("\nPress Enter when ready...")
    input()
    
    try:
        game = InstagramGame()
        game.play()
    except KeyboardInterrupt:
        print("\nGame automation stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
