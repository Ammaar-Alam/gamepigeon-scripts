import numpy as np
import cv2
import time
import gymnasium as gym
from gymnasium import spaces
import pyautogui
import subprocess
import os
from instagram_game import InstagramGame, find_iphone_mirror_window, move_iphone_mirror_to_corner, focus_iphone_window
import pytesseract
import uuid
from pynput.mouse import Button, Controller

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
        
        # Define action space: Small relative movements
        self.action_space = spaces.Box(
            low=-5, 
            high=5,
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
        
        # Start the game
        self.start_game()
        
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
        pyautogui.mouseUp()
        self.mouse_down = False
        time.sleep(0.2)
        
        # Click play and initialize
        self.game.click_play_again()
        print("Started game!")
        time.sleep(2)  # Increased delay to ensure the game has time to load
        
        # Move mouse to initial bar position and hold down button
        initial_x = self.game.window_x + self.game.WINDOW_WIDTH // 2
        initial_y = self.game.window_y + self.game.bar_y
        
        # Move mouse and ensure it's pressed
        pyautogui.moveTo(initial_x, initial_y)
        time.sleep(0.2)
        # Use pynput to hold down the mouse button
        try:
            mouse.press(Button.left)
            self.mouse_down = True
            time.sleep(0.2)
            print("Mouse button held down with pynput")
        except Exception as e:
            print(f"Error holding mouse down: {e}")
        
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
            try:
                mouse.release(Button.left)
                self.mouse_down = False
                time.sleep(0.1)
                print("Mouse button released with pynput")
            except Exception as e:
                print(f"Error releasing mouse button: {e}")
        
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
            try:
                mouse.press(Button.left)
                self.mouse_down = True
                time.sleep(0.1)
                print("Mouse button held down with pynput")
            except Exception as e:
                print(f"Error holding mouse down: {e}")
        
        # Get current screenshot
        screenshot = self.game.sct.grab(self.game.monitor)
        
        # Get current score
        current_score = self.get_score_from_image(screenshot)
        score_increased = current_score > self.last_score
        
        # Check if game is over first
        done = self.detect_game_over(screenshot)
        if done:
            if self.mouse_down:
                mouse.release(Button.left)
                self.mouse_down = False
                time.sleep(0.1)
                print("Mouse button released (game over)")
        else:
            # Get emoji position and move bar
            emoji_x = self.game.detect_emoji(screenshot)
            if emoji_x is not None:
                # Relative movement
                action_scaled = float(action[0])
                current_mouse_x, _ = pyautogui.position()
                new_mouse_x = current_mouse_x + action_scaled

                # Define the minimum and maximum X positions
                min_x = self.game.window_x + 20
                max_x = self.game.window_x + self.game.WINDOW_WIDTH - 20

                # Clip the new_mouse_x within the game window boundaries
                target_x = max(min_x, min(new_mouse_x, max_x))

                # Set screen_x directly to target_x
                screen_x = target_x
                screen_y = self.game.window_y + self.game.bar_y

                # Print debug statements
                print(f"Moving bar to screen coordinates: ({screen_x}, {screen_y})")

                try:
                    # Move the mouse while holding button
                    pyautogui.moveTo(screen_x, screen_y, _pause=False)
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
                mouse.release(Button.left)
                self.mouse_down = False
                print("Mouse button released (max steps)")
        
        # Update hits
        self.hits = self.last_score
        
        return observation, reward, done, False, {}
        
    def _calculate_reward(self, screenshot, done, emoji_x, score_increased):
        """Calculate reward based on game state."""
        reward = 0.0
        
        # Reward for increasing score
        if score_increased:
            reward += 1.0
            print(f"Reward +1.0 (Score increased to {self.last_score})")
        
        # Reward for aligning bar with emoji
        if emoji_x is not None:
            bar_x = pyautogui.position()[0] - self.game.window_x
            alignment = -abs(bar_x - emoji_x) / self.game.WINDOW_WIDTH
            reward += alignment  # Negative distance (closer is better)
        
        # Penalty for losing
        if done:
            penalty = -10.0 if self.last_score > 0 else -20.0
            reward += penalty
            print(f"Penalty {penalty} (Game Over)")
            
        return reward
        
    def close(self):
        """Clean up resources."""
        if self.mouse_down:
            mouse.release(Button.left)
            self.mouse_down = False
            time.sleep(0.1)
            print("Mouse button released (cleanup)")
        self.game.running = False
