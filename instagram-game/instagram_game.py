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
from pynput.mouse import Button, Controller
import os

# Configure pyautogui
pyautogui.FAILSAFE = False  # Disable failsafe since we have proper bounds checking
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0
pyautogui.DEFAULT_DURATION = 0  # Ensure instant movements

# Initialize mouse controller
mouse = Controller()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nExiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def find_iphone_mirror_window():
    """Find the iPhone Mirror window and get its position."""
    try:
        # Get all windows
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        # Look for iPhone Mirror window
        for window in windows:
            name = window.get(Quartz.kCGWindowName, '')
            owner = window.get(Quartz.kCGWindowOwnerName, '')
            
            if 'iPhone' in str(name) or 'iPhone' in str(owner):
                # Get window bounds
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
    
    # Verify window position
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
    """Hold down the left mouse button using AppleScript."""
    script = '''
    tell application "System Events"
        click down
    end tell
    '''
    try:
        subprocess.run(['osascript', '-e', script], check=True)
        print("Mouse button held down with AppleScript")
    except Exception as e:
        print(f"Error holding mouse down: {e}")

def release_mouse():
    """Release the left mouse button using AppleScript."""
    script = '''
    tell application "System Events"
        click up
    end tell
    '''
    try:
        subprocess.run(['osascript', '-e', script], check=True)
        print("Mouse button released with AppleScript")
    except Exception as e:
        print(f"Error releasing mouse: {e}")

class InstagramGame:
    def __init__(self):
        # Find iPhone Mirror window
        window_info = find_iphone_mirror_window()
        if not window_info:
            print("Could not find iPhone Mirror window!")
            sys.exit(1)
        
        # Window dimensions from actual window
        self.window_x, self.window_y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT = window_info
        print(f"Using window dimensions: x={self.window_x}, y={self.window_y}, w={self.WINDOW_WIDTH}, h={self.WINDOW_HEIGHT}")
        
        # Get screen scale factor
        self.scale_factor = 2  # Retina display
        print(f"Using scale factor: {self.scale_factor}")
        
        # Game area coordinates (relative to window)
        self.bar_y = int(self.WINDOW_HEIGHT * 0.93)  # Bar height
        self.play_again_y = int(self.WINDOW_HEIGHT * 0.92)  # Play again button
        
        # Game area boundaries (scaled coordinates)
        self.GAME_TOP = 150  # Start below scoreboard
        self.GAME_BOTTOM = self.WINDOW_HEIGHT - 50  # End above bottom bar
        print(f"Game area: y={self.GAME_TOP} to y={self.GAME_BOTTOM}")
        
        # Enhanced tracking state
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.velocity_x = 0
        self.velocity_y = 0
        self.stuck_counter = 0
        self.direction = 0  # -1: left, 1: right
        self.prediction_offset = 50  # Base prediction offset
        self.last_prediction = None
        self.last_score_y = None
        self.score = 0
        
        # Game state
        self.running = False
        self.game_started = False
        self.first_start = True
        self.game_start_time = None
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = {
            "top": self.window_y,
            "left": self.window_x,
            "width": self.WINDOW_WIDTH,
            "height": self.WINDOW_HEIGHT,
            "mon": 1  # Primary monitor
        }
        print(f"Capture area: {self.monitor}")
        
        # Previous image for frame differencing
        self.prev_img = None
        
        # Score tracking
        self.last_hit_time = None
        self.min_hit_interval = 0.5  # Minimum time between hits
    
    def scale_coordinates(self, x, y):
        """Scale coordinates from screenshot to screen space."""
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def update_tracking(self, x, y):
        """Update tracking with velocity calculation."""
        current_time = time.time()
        
        if self.last_x is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # Calculate velocity
                dx = x - self.last_x
                dy = y - self.last_y
                new_velocity_x = dx / dt
                new_velocity_y = dy / dt
                
                # Smooth velocity changes
                SMOOTHING = 0.7
                self.velocity_x = SMOOTHING * new_velocity_x + (1 - SMOOTHING) * self.velocity_x
                self.velocity_y = SMOOTHING * new_velocity_y + (1 - SMOOTHING) * self.velocity_y
                
                # Update direction
                if abs(dx) > 2:  # Ignore tiny movements
                    self.direction = 1 if dx > 0 else -1
                
                # Check if position is stuck
                if abs(dx) < 2 and abs(dy) < 2:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                
                # Track score by detecting upward bounces near bar
                if self.last_y and y < self.last_y:  # Moving upward
                    if self.last_y > self.GAME_BOTTOM - 100:  # Near bottom of screen
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
        """Predict emoji's next position."""
        if self.velocity_x == 0 and self.last_prediction is not None:
            return self.last_prediction
            
        # Base prediction time
        prediction_time = 0.1
        
        # Calculate predicted position
        predicted_x = x + int(self.velocity_x * prediction_time)
        
        # Add direction-based offset
        if self.direction != 0:
            # Scale offset with vertical position (more aggressive at bottom)
            vertical_factor = min((y - self.GAME_TOP) / (self.GAME_BOTTOM - self.GAME_TOP), 1.0)
            speed_factor = min(abs(self.velocity_x) / 1000, 1.0)
            offset = int(self.prediction_offset * vertical_factor * speed_factor) * self.direction
            predicted_x += offset
        
        # Ensure within bounds
        predicted_x = max(20, min(predicted_x, self.WINDOW_WIDTH - 20))
        
        # Store prediction
        self.last_prediction = predicted_x
        return predicted_x

    def detect_game_over(self, screenshot):
        """Improved game over detection."""
        if self.first_start:
            return False
            
        if not self.game_started or self.game_start_time is None:
            return False
            
        if (time.time() - self.game_start_time) < 1.0:  # Reduced from 2.0
            return False
        
        # Convert to numpy array and check for purple game over screen
        img = np.array(screenshot)
        purple_mask = (
            (img[..., 0] >= 120) & (img[..., 0] <= 170) &  # R
            (img[..., 1] >= 0) & (img[..., 1] <= 100) &    # G
            (img[..., 2] >= 180) & (img[..., 2] <= 255)    # B
        )
        
        # Calculate ratio in middle portion of screen
        height, width = purple_mask.shape[:2]
        middle_section = purple_mask[height//4:3*height//4, width//4:3*width//4]
        ratio = np.mean(middle_section)
        
        if ratio > 0.5:  # Increased threshold
            print(f"[GAME OVER] Detected purple screen (ratio: {ratio:.2f})")
            print(f"Final score: {self.score}")
            return True
            
        return False

    def detect_emoji(self, screenshot):
        """Detect emoji using color thresholds."""
        try:
            img = np.array(screenshot)

            # Convert to HSV color space
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define color range for the emoji (adjust as needed)
            lower_color = np.array([20, 100, 100])  # Example values
            upper_color = np.array([30, 255, 255])

            # Create a mask for the emoji color
            mask = cv2.inRange(hsv_img, lower_color, upper_color)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                x_center = x + w // 2
                y_center = y + h // 2

                # Scale coordinates
                x_scaled, y_scaled = self.scale_coordinates(x_center, y_center)

                # Update tracking
                self.update_tracking(x_scaled, y_scaled)
                predicted_x = self.predict_position(x_scaled, y_scaled)

                print(f"[DEBUG] Emoji detected at ({x_scaled}, {y_scaled})")
                return predicted_x
            else:
                print("Emoji not detected in color detection.")
                return None
        except Exception as e:
            print(f"Error in detect_emoji: {e}")
            return None

    def click_play_again(self):
        """Click play again button."""
        try:
            # Calculate absolute screen coordinates
            screen_x = self.window_x + self.WINDOW_WIDTH//2
            screen_y = self.window_y + self.play_again_y
            
            # Ensure within screen bounds
            screen_x = max(0, min(screen_x, self.sct.monitors[1]["width"]))
            screen_y = max(0, min(screen_y, self.sct.monitors[1]["height"]))
            
            pyautogui.moveTo(screen_x, screen_y, duration=0, _pause=False)
            pyautogui.click()
            
            self.game_started = True
            self.first_start = False
            self.game_start_time = time.time()
            
            # Reset tracking
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
        """Move bar to target position with smoother movement."""
        try:
            # Calculate absolute screen coordinates
            screen_x = self.window_x + target_x
            screen_y = self.window_y + self.bar_y

            # Add debugging statements
            print(f"Moving bar to screen coordinates: ({screen_x}, {screen_y})")

            # Ensure within screen bounds
            screen_x = max(0, min(screen_x, self.sct.monitors[1]["width"] - 1))
            screen_y = max(0, min(screen_y, self.sct.monitors[1]["height"] - 1))

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y, duration=0, _pause=False)
        except pyautogui.FailSafeException:
            print("FailSafeException triggered! Mouse movement stopped.")
        except Exception as e:
            print(f"Error moving bar: {e}")

    def play(self):
        """Main game loop."""
        print("Starting Instagram game automation...")
        print("Press Ctrl+C anytime to stop")
        
        while True:
            try:
                # Move and focus window
                window_info = move_iphone_mirror_to_corner()
                if window_info:
                    self.window_x, self.window_y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT = window_info
                    self.monitor.update({
                        "top": self.window_y,
                        "left": self.window_x,
                        "width": self.WINDOW_WIDTH,
                        "height": self.WINDOW_HEIGHT
                    })
                    print(f"Updated capture area: {self.monitor}")
                
                focus_iphone_window()
                
                self.running = True
                print("\nGame starting...")
                
                self.click_play_again()
                print("Started game!")
                
                # Move mouse to initial bar position
                initial_x = self.window_x + self.WINDOW_WIDTH // 2
                initial_y = self.window_y + self.bar_y
                pyautogui.moveTo(initial_x, initial_y, duration=0, _pause=False)
                
                # Hold down the left mouse button at the initial position
                try:
                    hold_mouse_down()
                    print("Mouse button held down")
                except Exception as e:
                    print(f"Error holding mouse down: {e}")
                
                while self.running:
                    # Fast screenshot
                    screenshot = self.sct.grab(self.monitor)
                    
                    if self.detect_game_over(screenshot):
                        self.running = False
                        break
                    
                    # Get emoji position and move bar
                    emoji_x = self.detect_emoji(screenshot)
                    if emoji_x is not None:
                        print(f"Moving bar to emoji_x: {emoji_x}")
                        self.move_bar(emoji_x)
                    else:
                        print("Emoji not detected.")
                
                # Release the left mouse button
                release_mouse()
                print("Mouse button released (game over)")
                
                print("\nPress Enter to play again, or Ctrl+C to quit...")
                input()
                
            except KeyboardInterrupt:
                print("\nExiting game...")
                sys.exit(0)
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                continue

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
