import mss.tools
import numpy as np
import pyautogui
import time
import sys
import signal
import cv2
from Cocoa import NSEvent
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
    kCGHIDEventTap,
    CGPointMake,
    CGEventSetIntegerValueField,
    kCGMouseEventPressure,
    kCGEventMouseMoved
)
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

def create_mouse_event(type, point, source=None):
    """Create a mouse event with proper source state."""
    if source is None:
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    event = CGEventCreateMouseEvent(source, type, point, kCGMouseButtonLeft)
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
        
        # Increase GAME_TOP to be well below the score display
        self.GAME_TOP = int(self.WINDOW_HEIGHT * 0.45)  # 45% from top
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
        self.last_move_time = None
        
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
        
        # Create debug window
        cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug View", 344, 764)
        
        # Create event source
        self.event_source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        
        # Initial mouse setup
        self.setup_initial_mouse_state()
    
    def setup_initial_mouse_state(self):
        """Set up initial mouse state using Quartz events."""
        try:
            initial_x = self.window_x + self.WINDOW_WIDTH // 2
            initial_y = self.window_y + self.bar_y
            point = CGPointMake(initial_x, initial_y)
            
            # Move to position
            move_event = create_mouse_event(kCGEventMouseMoved, point, self.event_source)
            CGEventPost(kCGHIDEventTap, move_event)
            time.sleep(0.1)
            
            # Press mouse button
            down_event = create_mouse_event(kCGEventLeftMouseDown, point, self.event_source)
            CGEventPost(kCGHIDEventTap, down_event)
            
            # Initial drag event
            drag_event = create_mouse_event(kCGEventLeftMouseDragged, point, self.event_source)
            CGEventPost(kCGHIDEventTap, drag_event)
            
            self.last_move_time = time.time()
            print("Initial mouse state set up")
            
        except Exception as e:
            print(f"Error in initial mouse setup: {e}")
    
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
        """Simplified emoji detection focusing only on the game area."""
        try:
            img = np.array(screenshot)
            
            # Create a mask for the game area
            game_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            game_mask[self.GAME_TOP:self.GAME_BOTTOM, :] = 255
            
            # Convert to HSV color space
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define color range for the emoji
            lower_color = np.array([20, 100, 100])
            upper_color = np.array([30, 255, 255])
            
            # Create a mask for the emoji color
            color_mask = cv2.inRange(hsv_img, lower_color, upper_color)
            
            # Apply game area mask
            final_mask = cv2.bitwise_and(color_mask, game_mask)
            
            # Show debug view
            debug_img = img.copy()
            
            # Draw game boundaries
            cv2.line(debug_img, (0, self.GAME_TOP), (self.WINDOW_WIDTH, self.GAME_TOP), (0, 0, 255), 2)
            cv2.line(debug_img, (0, self.GAME_BOTTOM), (self.WINDOW_WIDTH, self.GAME_BOTTOM), (0, 0, 255), 2)
            
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                x_center = x + w // 2
                y_center = y + h // 2
                
                # Draw detection
                cv2.circle(debug_img, (x_center, y_center), 5, (0, 255, 0), -1)
                cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 0), 2)
                
                x_scaled, y_scaled = self.scale_coordinates(x_center, y_center)
                self.update_tracking(x_scaled, y_scaled)
                predicted_x = self.predict_position(x_scaled, y_scaled)
                
                cv2.imshow("Debug View", debug_img)
                cv2.waitKey(1)
                
                return predicted_x
            
            cv2.imshow("Debug View", debug_img)
            cv2.waitKey(1)
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
            
            # Release mouse using Quartz
            point = CGPointMake(screen_x, screen_y)
            up_event = create_mouse_event(kCGEventLeftMouseUp, point, self.event_source)
            CGEventPost(kCGHIDEventTap, up_event)
            time.sleep(0.1)
            
            # Click using Quartz events
            down_event = create_mouse_event(kCGEventLeftMouseDown, point, self.event_source)
            CGEventPost(kCGHIDEventTap, down_event)
            time.sleep(0.1)
            up_event = create_mouse_event(kCGEventLeftMouseUp, point, self.event_source)
            CGEventPost(kCGHIDEventTap, up_event)
            
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
            
            # Reset mouse state
            self.setup_initial_mouse_state()
            
        except Exception as e:
            print(f"Error clicking play: {e}")

    def move_bar(self, target_x):
        """Move bar using Quartz events with proper timing."""
        try:
            current_time = time.time()
            
            # Add small delay between moves
            if self.last_move_time and current_time - self.last_move_time < 0.016:  # ~60 FPS
                return
            
            screen_x = self.window_x + target_x
            screen_y = self.window_y + self.bar_y
            
            screen_x = max(0, min(screen_x, self.sct.monitors[1]["width"] - 1))
            screen_y = max(0, min(screen_y, self.sct.monitors[1]["height"] - 1))
            
            point = CGPointMake(screen_x, screen_y)
            
            # Create drag event
            drag_event = create_mouse_event(kCGEventLeftMouseDragged, point, self.event_source)
            CGEventPost(kCGHIDEventTap, drag_event)
            
            self.last_move_time = current_time
            print(f"Bar moved to: ({screen_x}, {screen_y})")
            
        except Exception as e:
            print(f"Error moving bar: {e}")
            # Reset mouse state on error
            self.setup_initial_mouse_state()
            
    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\nINSTRUCTIONS:")
    print("1. Make sure iPhone Mirroring is running")
    print("2. The Instagram game should be ready")
    print("3. Press Enter to start - window will be moved automatically")
    print("\nPress Enter when ready...")
    input()
    
    try:
        game = InstagramGame()
    except KeyboardInterrupt:
        print("\nGame automation stopped by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
