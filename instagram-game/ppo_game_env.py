import numpy as np
import cv2
import time
import gymnasium as gym
from gymnasium import spaces
import pyautogui
import subprocess
import os
from instagram_game import InstagramGame, find_iphone_mirror_window, move_iphone_mirror_to_corner, focus_iphone_window, create_mouse_event
import pytesseract
import uuid
from pynput.mouse import Button, Controller
import Quartz

mouse = Controller()

class GameEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        print("\nINSTRUCTIONS:")
        print("1. Make sure iPhone Mirroring is running")
        print("2. The Instagram game should be ready")
        print("3. Press Enter to start - window will be moved automatically")
        print("\nPress Enter when ready...")
        input()
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Initialize the base game
        self.game = InstagramGame()
        
        # Define action space: Relative movements (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, 
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space: Game screen + ball position + ball velocity
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            'ball_pos': spaces.Box(low=0, high=max(self.game.WINDOW_WIDTH, self.game.WINDOW_HEIGHT), shape=(2,), dtype=np.float32),
            'ball_vel': spaces.Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
        })
        
        self.current_score = 0
        self.episode_steps = 0
        self.max_steps = 1000
        self.mouse_down = False
        self.first_start = True
        self.last_score = 0
        self.hits = 0
        self.last_bar_x = None
        
    def _press_mouse(self):
        """Press and hold left mouse button using Quartz."""
        try:
            point = Quartz.CGPointMake(mouse.position[0], mouse.position[1])
            event = create_mouse_event(Quartz.kCGEventLeftMouseDown, point)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            
            # Create and post a dragged event to maintain the press
            drag_event = create_mouse_event(Quartz.kCGEventLeftMouseDragged, point)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, drag_event)
            
            self.mouse_down = True
            time.sleep(0.1)
            print("Mouse button held down with Quartz")
        except Exception as e:
            print(f"Error holding mouse down: {e}")
            
    def _release_mouse(self):
        """Release left mouse button using Quartz."""
        try:
            point = Quartz.CGPointMake(mouse.position[0], mouse.position[1])
            event = create_mouse_event(Quartz.kCGEventLeftMouseUp, point)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            self.mouse_down = False
            time.sleep(0.1)
            print("Mouse button released with Quartz")
        except Exception as e:
            print(f"Error releasing mouse: {e}")
        
    def start_game(self):
        """Initialize game state."""
        if not self.first_start:
            # Move and focus window
            window_info = move_iphone_mirror_to_corner()
            if window_info:
                self.game.window_x, self.game.window_y, self.game.WINDOW_WIDTH, self.game.WINDOW_HEIGHT = window_info
                self.game.monitor.update({
                    "top": self.game.window_y,
                    "left": self.game.window_x,
                    "width": self.game.WINDOW_WIDTH,
                    "height": self.game.WINDOW_HEIGHT
                })
                print(f"Updated capture area: {self.game.monitor}")
        
        focus_iphone_window()
        time.sleep(0.2)  # Wait for window focus
        
        self.game.running = True
        print("\nGame starting...")
        
        # Release mouse first
        self._release_mouse()
        time.sleep(0.2)
        
        # Click play and initialize
        self.game.click_play_again()
        print("Started game!")
        time.sleep(2)  # Increased delay to ensure the game has time to load
        
        # Move mouse to initial bar position and hold down button
        initial_x = self.game.window_x + self.game.WINDOW_WIDTH // 2
        initial_y = self.game.window_y + self.game.bar_y
        
        # Move mouse and ensure it's pressed
        point = Quartz.CGPointMake(initial_x, initial_y)
        move_event = create_mouse_event(Quartz.kCGEventMouseMoved, point)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_event)
        time.sleep(0.2)
        
        self._press_mouse()
        self.last_bar_x = initial_x
        
        self.first_start = False
        self.last_score = 0
        
    def get_score_from_image(self, screenshot):
        """Extract score from the screenshot using OCR."""
        try:
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Get the score region; adjust these values based on actual position
            height, width = img.shape[:2]
            # Adjust the coordinates as needed to tightly crop the score area
            y_start = int(height * 0.05)
            y_end = int(height * 0.15)
            x_start = int(width * 0.3)
            x_end = int(width * 0.7)
            score_region = img[y_start:y_end, x_start:x_end]
            
            # Convert to grayscale
            gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                blockSize=11, 
                C=2
            )
            
            # Optionally, apply dilation to enhance features
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            # OCR to get text
            text = pytesseract.image_to_string(
                thresh, 
                config='--psm 7 -c tessedit_char_whitelist=0123456789'
            )
            
            # Extract number
            score = ''.join(filter(str.isdigit, text))
            if score:
                current_score = int(score)
                if current_score != self.last_score:
                    print(f"Score updated: {current_score}")
                    self.last_score = current_score
                return current_score
        except Exception as e:
            print(f"Error reading score: {e}")
        return self.last_score
        
    def verify_game_started(self):
        """Verify game has properly started by checking for emoji."""
        max_checks = 10
        for attempt in range(max_checks):
            screenshot = self.game.sct.grab(self.game.monitor)
            emoji_x = self.game.detect_emoji(screenshot)
            if emoji_x is not None:
                print("Game started successfully.")
                return True
            else:
                print(f"Attempt {attempt+1}/{max_checks}: Emoji not detected, retrying...")
                time.sleep(0.5)  # Wait before retrying
        print("Game didn't start properly after multiple attempts.")
        return False
        
    def preprocess_observation(self, screenshot):
        """Convert screenshot to neural network input."""
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        
        # Add channel dimension and normalize
        processed = resized[..., np.newaxis].astype(np.float32) / 255.0
        
        return processed
        
    def detect_game_over(self, screenshot):
        """Detect game over using purple color."""
        img = np.array(screenshot)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define purple color range in HSV
        lower_purple = np.array([270/2, 50, 50])  # HSV values for purple
        upper_purple = np.array([290/2, 255, 255])
        
        # Create mask for purple color
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Calculate percentage of purple pixels
        purple_ratio = np.sum(purple_mask > 0) / (purple_mask.shape[0] * purple_mask.shape[1])
        
        # Debug output
        if purple_ratio > 0.4:
            print(f"\n[GAME OVER] Detected purple screen (ratio: {purple_ratio:.2f})")
            print(f"Final score: {self.last_score}")
            return True
            
        return False
        
    def reset(self, seed=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Release mouse if it was held down
        if self.mouse_down:
            self._release_mouse()
        
        # Restart game and verify it started
        max_attempts = 3
        for attempt in range(max_attempts):
            self.start_game()
            if self.verify_game_started():
                break
            if attempt < max_attempts - 1:
                time.sleep(1)  # Wait before retrying
        
        # Get initial observation
        screenshot = self.game.sct.grab(self.game.monitor)
        
        # Get ball position and velocity
        emoji_x = self.game.detect_emoji(screenshot)
        if emoji_x is not None:
            ball_pos = np.array([emoji_x, self.game.last_y or 0], dtype=np.float32)
            ball_vel = np.array([self.game.velocity_x, 0], dtype=np.float32)
        else:
            ball_pos = np.array([self.game.WINDOW_WIDTH/2, 0], dtype=np.float32)
            ball_vel = np.array([0, 0], dtype=np.float32)
        
        observation = {
            'screen': self.preprocess_observation(screenshot),
            'ball_pos': ball_pos,
            'ball_vel': ball_vel
        }
        
        self.episode_steps = 0
        self.current_score = 0
        self.last_score = 0
        self.hits = 0
        
        return observation, {}
        
    def step(self, action):
        """Execute action and return new state."""
        self.episode_steps += 1
        
        # Ensure mouse is held down
        if not self.mouse_down:
            self._press_mouse()
        
        # Get current screenshot
        screenshot = self.game.sct.grab(self.game.monitor)
        
        # Get current score
        current_score = self.get_score_from_image(screenshot)
        score_increased = current_score > self.last_score
        
        # Check if game is over first
        done = self.detect_game_over(screenshot)
        if done:
            if self.mouse_down:
                self._release_mouse()
        else:
            # Get emoji position and move bar
            emoji_x = self.game.detect_emoji(screenshot)
            if emoji_x is not None:
                # Scale action from [-1, 1] to actual pixel movement
                max_movement = 20  # Maximum pixels to move per step
                action_scaled = float(action[0]) * max_movement
                
                # Get current bar position
                current_x = self.last_bar_x if self.last_bar_x is not None else mouse.position[0]
                new_x = current_x + action_scaled

                # Define the minimum and maximum X positions
                min_x = self.game.window_x + 20
                max_x = self.game.window_x + self.game.WINDOW_WIDTH - 20

                # Clip the new_x within the game window boundaries
                target_x = max(min_x, min(new_x, max_x))

                # Move the mouse while maintaining Y position
                try:
                    point = Quartz.CGPointMake(target_x, self.game.window_y + self.game.bar_y)
                    # Create and post a mouse moved event
                    move_event = create_mouse_event(Quartz.kCGEventMouseMoved, point)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_event)
                    
                    # Create and post a dragged event to maintain the press
                    drag_event = create_mouse_event(Quartz.kCGEventLeftMouseDragged, point)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, drag_event)
                    
                    self.last_bar_x = target_x
                except Exception as e:
                    print(f"Error moving bar: {e}")
            else:
                print("Emoji not detected.")
        
        # Get new observation
        new_screenshot = self.game.sct.grab(self.game.monitor)
        
        # Get ball position and velocity
        emoji_x = self.game.detect_emoji(new_screenshot)
        if emoji_x is not None:
            ball_pos = np.array([emoji_x, self.game.last_y or 0], dtype=np.float32)
            ball_vel = np.array([self.game.velocity_x, 0], dtype=np.float32)
        else:
            ball_pos = np.array([self.game.WINDOW_WIDTH/2, 0], dtype=np.float32)
            ball_vel = np.array([0, 0], dtype=np.float32)
        
        # Normalize observations
        max_pos = np.array([self.game.WINDOW_WIDTH, self.game.WINDOW_HEIGHT], dtype=np.float32)
        ball_pos = ball_pos / max_pos
        max_vel = 1000  # or use actual max velocity
        ball_vel = ball_vel / max_vel

        # Create observation dictionary
        observation = {
            'screen': self.preprocess_observation(new_screenshot),
            'ball_pos': ball_pos,
            'ball_vel': ball_vel
        }
        
        # Calculate reward
        reward = self._calculate_reward(new_screenshot, done, emoji_x, score_increased)
        
        # Check if max steps reached
        if self.episode_steps >= self.max_steps:
            done = True
            if self.mouse_down:
                self._release_mouse()
        
        # Update hits
        self.hits = self.last_score
        
        return observation, reward, done, False, {}
        
    def _calculate_reward(self, screenshot, done, emoji_x, score_increased):
        """Calculate reward based on game state."""
        reward = 0.0
        
        # Reward for increasing score
        if score_increased:
            reward += 10.0  # Increased reward for successful hits
            print(f"Reward +10.0 (Score increased to {self.last_score})")
        
        # Reward for aligning bar with emoji
        if emoji_x is not None and self.last_bar_x is not None:
            # Calculate normalized distance between bar and emoji
            max_distance = self.game.WINDOW_WIDTH
            distance = abs(self.last_bar_x - (self.game.window_x + emoji_x))
            normalized_distance = distance / max_distance
            
            # Exponential reward based on proximity
            proximity_reward = np.exp(-5 * normalized_distance) - 0.5
            reward += proximity_reward
            
            # Additional reward for perfect alignment
            if distance < 10:
                reward += 1.0
        
        # Penalty for losing
        if done:
            base_penalty = -20.0
            score_factor = max(0, min(self.last_score, 10)) / 10.0  # Scale based on score (up to 10)
            penalty = base_penalty * (1 - score_factor)  # Reduce penalty based on score
            reward += penalty
            print(f"Penalty {penalty:.1f} (Game Over)")
            
        return reward
        
    def close(self):
        """Clean up resources."""
        if self.mouse_down:
            self._release_mouse()
        self.game.running = False
