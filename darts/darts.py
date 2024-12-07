import pyautogui
import cv2
import numpy as np
import time
from PIL import ImageGrab
import subprocess
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import math

# Prevent mouse from going to screen corners
pyautogui.FAILSAFE = True

# Fixed X-offsets from center for each number (based on our working triple-20)
# Positive is right, negative is left
NUMBER_OFFSETS = {
    20: 0,      # Center (our reference)
    1: 15,      # Slightly right
    18: 25,     # More right
    4: 30,      # Far right
    13: 25,     # More right
    6: 15,      # Slightly right
    10: 0,      # Center
    15: -15,    # Slightly left
    2: -25,     # More left
    17: -30,    # Far left
    3: -25,     # More left
    19: -15,    # Slightly left
    7: 0,       # Center
    16: 15,     # Slightly right
    8: 25,      # More right
    11: 30,     # Far right
    14: 25,     # More right
    9: 15,      # Slightly right
    12: 0,      # Center
    5: -15,     # Slightly left
}

# Y-offsets from triple position (positive is down, negative is up)
VERTICAL_OFFSETS = {
    'single': 40,    # Below triple
    'triple': 0,     # Reference point
    'double': -40    # Above triple
}

class DartsAI:
    def __init__(self):
        self.model = self._build_model()
        self.training_data = []
        self.current_adjustment = 0
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def _build_model(self):
        """
        Build a simple model that only learns Y-offset adjustments.
        Input: [target_number, current_y_offset]
        Output: [y_adjustment]
        """
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(2,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='tanh')  # Output small adjustments
        ])
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def save_model(self, filename='darts_model.h5'):
        """Save the trained model."""
        self.model.save(os.path.join('models', filename))
    
    def load_model(self, filename='darts_model.h5'):
        """Load a trained model."""
        model_path = os.path.join('models', filename)
        if os.path.exists(model_path):
            self.model = models.load_model(model_path)
            return True
        return False
    
    def predict_adjustment(self, target_number, current_y_offset):
        """Predict Y-offset adjustment."""
        input_data = np.array([[target_number, current_y_offset]])
        self.current_adjustment = float(self.model.predict(input_data, verbose=0)[0][0])
        return self.current_adjustment
    
    def train(self, target_number, y_offset, success_score):
        """Train the model with a throw result."""
        self.training_data.append({
            'target': target_number,
            'y_offset': y_offset,
            'score': success_score
        })
        
        if len(self.training_data) >= 3:  # Train in very small batches
            X = np.array([[d['target'], d['y_offset']] for d in self.training_data])
            y = np.array([d['score'] for d in self.training_data])
            
            self.model.fit(X, y, epochs=3, verbose=0)
            self.training_data = []  # Clear after training

def move_iphone_mirror_to_corner():
    """
    Move iPhone Mirroring window to bottom right corner and focus it.
    """
    script = """
    tell application "System Events"
        tell application process "iPhone Mirroring"
            set frontmost to true
            delay 0.5
            click menu item "Bottom Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1
        end tell
    end tell
    """
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.stderr:
        print(f"Error: {result.stderr}")
    if result.stdout:
        print(f"Output: {result.stdout}")

def focus_iphone_window():
    """Focus the iPhone Mirroring window."""
    script = """
    tell application "System Events"
        tell application process "iPhone Mirroring"
            set frontmost to true
        end tell
    end tell
    """
    subprocess.run(["osascript", "-e", script], capture_output=True)

def get_dart_and_target_positions(target_number, multiplier_type, y_offset_multiplier=1.0):
    """
    Calculate the starting dart position and target position using fixed reference points.
    """
    # Known screen resolution: 1800x1169
    screen_width = 1800
    screen_height = 1169
    
    # iPhone mirror window dimensions when in corner
    window_width = 400
    window_height = 800
    
    # Calculate window position
    window_x = screen_width - window_width
    window_y = screen_height - window_height
    
    # Base center X coordinate (from our working triple-20)
    center_x = window_x + (window_width // 2) + 20
    
    # Apply number-specific X offset
    target_x = center_x + NUMBER_OFFSETS[target_number]
    
    # Dart starts at the bottom center of the window
    dart_x = window_x + (window_width // 2) + 20
    dart_y = window_y + int(window_height * 0.75)  # Keep the same start height
    
    # Use fixed reference end Y position (like in darts_feedback_ai.py)
    base_end_y = 419  # Known good reference point for triple zone
    
    # Apply vertical offset based on multiplier
    vertical_offset = VERTICAL_OFFSETS[multiplier_type]
    
    # Apply AI adjustment (scaled down to prevent extreme adjustments)
    ai_adjustment = int(500 * y_offset_multiplier)  # Maximum 50 pixel adjustment
    
    # Calculate final end Y position
    end_y = base_end_y + vertical_offset + ai_adjustment
    
    return (dart_x, dart_y), (target_x, end_y)

def ease_out_quad(x):
    """
    Quadratic easing function for smoother motion.
    """
    return 1 - (1 - x) * (1 - x)

def throw_dart(start_pos, end_pos):
    """
    Simulate a dart throw with a smooth drag motion.
    """
    print(f"Throwing dart from {start_pos} to {end_pos}")
    
    # Focus iPhone window before throw
    focus_iphone_window()
    time.sleep(0.5)  # Wait for window focus
    
    # Move to start position
    print("Moving to start position...")
    pyautogui.moveTo(start_pos[0], start_pos[1], duration=0.1)
    
    # Small pause to ensure we're at start position
    time.sleep(0.1)
    
    # Perform the drag with custom easing
    print("Performing drag motion...")
    pyautogui.dragTo(
        end_pos[0], 
        end_pos[1], 
        duration=0.17,  # Keep the smooth speed
        tween=ease_out_quad,  # Keep the smooth motion
        button='left'
    )
    
    # Wait for throw animation
    time.sleep(2)

def train_ai():
    """
    Training mode for the AI.
    """
    ai = DartsAI()
    
    print("\nStarting AI training mode...")
    print("\nAvailable numbers:", list(NUMBER_OFFSETS.keys()))
    print("Available multipliers:", list(VERTICAL_OFFSETS.keys()))
    
    while True:
        try:
            target_number = int(input("\nEnter target number (or 0 to exit): "))
            if target_number == 0:
                break
            if target_number not in NUMBER_OFFSETS and target_number != 25:
                print("Invalid number. Please choose from:", list(NUMBER_OFFSETS.keys()))
                continue
            
            print("\nMultipliers:")
            multipliers = list(VERTICAL_OFFSETS.keys())
            for i, mult in enumerate(multipliers, 1):
                print(f"{i}. {mult}")
            multiplier_idx = int(input("Choose multiplier (1-3): ")) - 1
            if multiplier_idx not in range(len(multipliers)):
                print("Invalid multiplier")
                continue
            
            multiplier = multipliers[multiplier_idx]
            
            print(f"\nTraining for {multiplier} {target_number}")
            print("Rate each throw from 0-10 (10 = perfect hit)")
            print("Press Ctrl+C to choose a different target")
            
            current_y_offset = 0
            
            while True:
                # Get current Y offset multiplier from model
                y_offset = ai.predict_adjustment(target_number, current_y_offset)
                print(f"\nCurrent Y adjustment: {y_offset:.4f}")
                
                # Get positions with AI adjustment
                start_pos, end_pos = get_dart_and_target_positions(target_number, multiplier, y_offset)
                
                # Throw dart
                throw_dart(start_pos, end_pos)
                
                print("\nSwitch back to VSCode to rate the throw")
                # Get feedback
                score = float(input("Rate the throw (0-10): ")) / 10.0
                
                # Train model
                ai.train(target_number, y_offset, score)
                current_y_offset = y_offset
                
                # Save periodically
                ai.save_model()
                
        except KeyboardInterrupt:
            print("\nSwitching targets...")
            continue
        except ValueError:
            print("Invalid input")
            continue
    
    print("\nTraining completed")
    ai.save_model()

def automate_darts(num_throws=3):
    """
    Main function to automate the dart game.
    """
    print("Starting darts automation...")
    
    # Initialize AI
    ai = DartsAI()
    if not ai.load_model():
        print("No trained model found, using base settings")
    
    # Get target from user
    print("\nAvailable numbers:", list(NUMBER_OFFSETS.keys()))
    target_number = int(input("Enter target number: "))
    if target_number not in NUMBER_OFFSETS and target_number != 25:
        print("Invalid number, defaulting to 20")
        target_number = 20
    
    print("\nMultipliers:")
    multipliers = list(VERTICAL_OFFSETS.keys())
    for i, mult in enumerate(multipliers, 1):
        print(f"{i}. {mult}")
    multiplier_idx = int(input("Choose multiplier (1-3): ")) - 1
    if multiplier_idx not in range(len(multipliers)):
        print("Invalid multiplier, defaulting to triple")
        multiplier_idx = 2
    
    multiplier = multipliers[multiplier_idx]
    
    current_y_offset = 0
    
    for throw_num in range(num_throws):
        # Get AI adjustment if model exists
        y_offset = ai.predict_adjustment(target_number, current_y_offset) if ai.load_model() else 1.0
        print(f"\nCurrent Y adjustment: {y_offset:.4f}")
        
        # Get positions with AI adjustment
        start_pos, end_pos = get_dart_and_target_positions(target_number, multiplier, y_offset)
        
        # Perform the throw
        throw_dart(start_pos, end_pos)
        
        print(f"Throw {throw_num + 1}/{num_throws} completed!")
        current_y_offset = y_offset
        
        # Wait between throws
        time.sleep(2)
    
    print("All throws completed!")

if __name__ == "__main__":
    # Safety prompt
    print("Please ensure:")
    print("1. The iPhone mirror window is visible")
    print("2. The darts game is ready to play")
    print("3. The dartboard is clearly visible")
    print("\nOptions:")
    print("1. Normal mode (using trained AI if available)")
    print("2. Training mode")
    
    choice = input("Enter mode (1 or 2): ").strip()
    
    try:
        # Move window to corner first
        print("\nPositioning iPhone window...")
        move_iphone_mirror_to_corner()
        time.sleep(1)  # Wait for window to move
        
        # Then start selected mode
        if choice == "2":
            train_ai()
        else:
            automate_darts()
    except KeyboardInterrupt:
        print("\nAutomation stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
