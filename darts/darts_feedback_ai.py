import asyncio
import websockets
import json
import numpy as np
import os
import pyautogui
import time
import subprocess
import math
from datetime import datetime
import re
from ppo_agent import DartsPPO

# Prevent mouse from going to screen corners
pyautogui.FAILSAFE = True

def parse_coords(coord_str):
    """Parse coordinate string like '(162, 94)' into x,y tuple"""
    match = re.match(r'\((\d+),\s*(\d+)\)', coord_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_score_text(score_text):
    """Parse score text like '20 x3 = 60' into number and multiplier."""
    try:
        if isinstance(score_text, str) and 'x' in score_text:
            number = int(score_text.split('x')[0].strip())
            multiplier = int(score_text.split('=')[0].split('x')[1].strip())
            return number, multiplier
    except:
        pass
    return None, None

class DartsFeedbackAI:
    def __init__(self, mode="training"):
        # Set AI mode (training or competing)
        self.mode = mode
        print(f"AI initialized in {mode} mode")

        # Map multiplier strings to numbers
        self.multiplier_map = {
            'single': 1,
            'double': 2,
            'triple': 3
        }

        # Our known working triple-20 throw
        self.reference_throw = {
            'start_x': 1620,
            'start_y': 969,
            'end_x': 1620,
            'end_y': 469,  # Lower end_y for better accuracy
            'duration': 0.17  # Fixed duration that works
        }

        # Initialize PPO agent
        # State: [target_x, target_y, target_multiplier, prev_landing_x, prev_landing_y, prev_score]
        # Action: [x_adjust, y_adjust]  # Only learn adjustments from reference throw
        self.agent = DartsPPO(state_dim=6, action_dim=2)
        
        # Load best model if in competing mode
        if mode == "competing" and os.path.exists('models/best_model.pth'):
            self.agent.load_model('models/best_model.pth')

        # Initialize state
        self.last_throw = None
        self.current_target = None
        self.last_target_coords = None
        self.best_throws = {}  # Store best throws per target
        self.current_throw_params = None
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None

        # Screen to board coordinate scaling
        self.x_scale = 2.0
        self.y_scale = 2.0

        # Maximum adjustments
        self.MAX_ADJUST = 200

        # Screen boundaries
        self.screen_width, self.screen_height = pyautogui.size()

        # Create directories
        for dir_path in ['models', 'throw_history']:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Load throw history
        self.load_history()

        # Training stats
        self.episode_rewards = []
        self.training_step = 0
        self.best_reward = -float('inf')

    def get_state(self, target_coords, target_multiplier):
        """Convert current situation to state vector for PPO."""
        # Default previous landing to target coords if no previous throw
        prev_landing = self.last_throw if self.last_throw else target_coords
        prev_score = 0.0 if not hasattr(self, 'last_score') else self.last_score
        
        return np.array([
            target_coords[0] / 350.0,  # Normalize coordinates
            target_coords[1] / 350.0,
            target_multiplier / 3.0,    # Normalize multiplier
            prev_landing[0] / 350.0,
            prev_landing[1] / 350.0,
            prev_score
        ], dtype=np.float32)

    def action_to_throw_params(self, action):
        """Convert PPO action to throw parameters."""
        # Scale actions to adjustments from reference throw
        x_adjust = np.clip(action[0] * 100, -160, 160)  # Max adjustment of 160 pixels
        y_adjust = np.clip(action[1] * 100, -160, 160)

        # Calculate end positions relative to reference throw
        end_x = int(self.reference_throw['end_x'] + x_adjust)
        end_y = int(self.reference_throw['end_y'] + y_adjust)

        return {
            'start_x': self.reference_throw['start_x'],
            'start_y': self.reference_throw['start_y'],
            'end_x': end_x,
            'end_y': end_y,
            'duration': self.reference_throw['duration']  # Use fixed duration
        }

    def calculate_reward(self, target_coords, landing_coords, target_multiplier, hit_multiplier, hit_number, target_number):
        """Calculate reward for the throw."""
        # Calculate distance error
        error_x = target_coords[0] - landing_coords[0]
        error_y = target_coords[1] - landing_coords[1]
        distance_error = math.sqrt(error_x**2 + error_y**2)
        
        # Base reward based on distance (exponential falloff)
        base_score = math.exp(-distance_error / 50)
        # Cap base score at 1.0
        base_score = min(base_score, 1.0)
        
        # Bonus for hitting correct multiplier and number
        if hit_number == target_number and hit_multiplier == self.multiplier_map[target_multiplier]:
            base_score = min(base_score * 1.5, 1.0)  # Apply bonus but cap at 1.0
        
        return base_score

    def load_history(self):
        """Load throw history."""
        try:
            with open('throw_history/throws.json', 'r') as f:
                self.throw_history = json.load(f)
        except FileNotFoundError:
            self.throw_history = {}

    def save_history(self):
        """Save throw history to file."""
        with open('throw_history/throws.json', 'w') as f:
            json.dump(self.throw_history, f, indent=2)

    def get_target_key(self, number, multiplier):
        """Convert number and multiplier to consistent target key format."""
        return f"{number}_{multiplier}"

    def execute_throw(self, target_number, target_multiplier):
        """Execute throw using PPO agent."""
        self.current_target = (target_number, target_multiplier)
        target_key = self.get_target_key(target_number, target_multiplier)

        # Get target coordinates from dartboard UI coordinates
        dartboard_coords = {
            # Top section (20)
            '20_single': [175, 69],
            '20_double': [175, 41],
            '20_triple': [175, 100],

            # Right section (clockwise)
            '1_single': [205, 75],
            '1_double': [215, 45],
            '1_triple': [200, 105],

            '18_single': [237, 90],
            '18_double': [255, 65],
            '18_triple': [220, 114],

            '4_single': [260, 117],
            '4_double': [288, 95],
            '4_triple': [237, 129], 

            '13_single': [277, 143],
            '13_double': [305, 133],
            '13_triple': [248, 152],

            '6_single': [280, 175],
            '6_double': [310, 175],
            '6_triple': [250, 175],

            '10_single': [275, 205],
            '10_double': [300, 215],
            '10_triple': [248, 198],

            '15_single': [260, 235],
            '15_double': [287, 254],
            '15_triple': [237, 221],

            '2_single': [240, 260],
            '2_double': [255, 285],
            '2_triple': [220, 235],

            '17_single': [205, 275],
            '17_double': [218, 305],
            '17_triple': [200, 250],

            '3_single': [175, 285],
            '3_double': [175, 315],
            '3_triple': [175, 250],

            # Left section (mirrored coordinates)
            '5_single': [145, 75],
            '5_double': [135, 45],
            '5_triple': [154, 103],

            '12_single': [112, 87],
            '12_double': [95, 65],
            '12_triple': [130, 115],

            '9_single': [90, 110],
            '9_double': [65, 95],
            '9_triple': [115, 130],

            '14_single': [75, 145],
            '14_double': [45, 135],
            '14_triple': [100, 150],

            '11_single': [75, 175],
            '11_double': [40, 175],
            '11_triple': [100, 175],

            '8_single': [75, 210],
            '8_double': [45, 220],
            '8_triple': [103, 200],

            '16_single': [90, 230],
            '16_double': [65, 255],
            '16_triple': [115, 220],

            '7_single': [115, 260],
            '7_double': [95, 285],
            '7_triple': [130, 235],

            '19_single': [145, 275],
            '19_double': [130, 305],
            '19_triple': [150, 250],

            # Center
            'bullseye': [175, 175],
            'double_bullseye': [175, 175]
        }

        # Get target coordinates
        target_coords = dartboard_coords.get(target_key, [175, 175])

        # In competing mode, use best throw if available
        if self.mode == "competing" and target_key in self.best_throws:
            throw_params = self.best_throws[target_key]
            print(f"Using best throw for {target_key}")
        else:
            # Get state and action from PPO agent
            state = self.get_state(target_coords, self.multiplier_map[target_multiplier])
            action, log_prob = self.agent.select_action(state)
            throw_params = self.action_to_throw_params(action)

            # Store for training update
            self.last_state = state
            self.last_action = action
            self.last_log_prob = log_prob

        print(f"Executing throw with parameters: {throw_params}")

        # Execute the throw
        try:
            self.focus_iphone_window()
            time.sleep(0.5)

            # Move to start position
            pyautogui.moveTo(
                throw_params['start_x'],
                throw_params['start_y'],
                duration=0.1
            )
            time.sleep(0.1)

            # Perform throw - use simple dragTo like original code
            pyautogui.dragTo(
                throw_params['end_x'],
                throw_params['end_y'],
                duration=throw_params['duration'],
                button='left'
            )

            self.current_throw_params = throw_params

        except Exception as e:
            print(f"Error executing throw: {e}")

        return throw_params

    def train(self, target_coords, landing_coords, hit_number, hit_multiplier):
        """Train PPO agent on throw result."""
        if self.mode != "training":
            return

        # Check if we have all required data
        if (self.last_state is None or 
            self.last_action is None or 
            self.last_log_prob is None):
            print("Missing required training data")
            return

        # Calculate reward
        reward = self.calculate_reward(
            target_coords, 
            landing_coords,
            self.current_target[1],  # target_multiplier
            hit_multiplier,
            hit_number,
            self.current_target[0]   # target_number
        )

        # Get next state
        next_state = self.get_state(target_coords, self.multiplier_map[self.current_target[1]])

        # Store experience
        self.agent.store_experience(
            self.last_state,
            self.last_action,
            reward,
            next_state,
            self.last_log_prob
        )

        # Update training stats
        self.episode_rewards.append(reward)
        self.training_step += 1

        # Train the agent every N steps
        if self.training_step % 5 == 0:  # Train more frequently
            print("\nTraining PPO agent...")
            self.agent.train(batch_size=min(16, len(self.agent.experiences)), epochs=5)
            
            # Calculate average reward
            avg_reward = sum(self.episode_rewards[-5:]) / len(self.episode_rewards[-5:])
            print(f"Average reward over last 5 throws: {avg_reward:.2f}")
            
            # Save if improved
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.agent.save_model('models/best_model.pth')
                print(f"New best model saved with average reward: {avg_reward:.2f}")

        # Update state
        self.last_throw = landing_coords
        self.last_score = reward

        return reward

    def focus_iphone_window(self):
        """Focus the iPhone Mirroring window."""
        script = """
        tell application "System Events"
            tell application process "iPhone Mirroring"
                set frontmost to true
            end tell
        end tell
        """
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.5)

    def move_iphone_mirror_to_corner(self):
        """Move iPhone Mirroring window to bottom right corner."""
        try:
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
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not move iPhone window: {e}")

async def handle_client(websocket, path):
    """Handle WebSocket connection from feedback UI."""
    mode = "training"
    if "?" in path and "mode=competing" in path.split("?")[1]:
        mode = "competing"
    
    ai = DartsFeedbackAI(mode=mode)
    print(f"Client connected in {mode} mode")

    ai.move_iphone_mirror_to_corner()

    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received message: {data}")

            if data['type'] == 'throw_request':
                target = data['data']
                print(f"\nAiming for {target['multiplier']} {target['number']}")

                throw_params = ai.execute_throw(
                    int(target['number']),
                    target['multiplier']
                )
                print(f"Throw executed with parameters: {throw_params}")

            elif data['type'] == 'landing_data' and mode == "training":
                target_x, target_y = parse_coords(data['data']['target'])
                landing_x, landing_y = parse_coords(data['data']['landing'])
                throw_failed = data['data'].get('throw_failed', False)

                if all([target_x, target_y, landing_x, landing_y]) and ai.current_target:
                    hit_number = 0
                    hit_multiplier = 0
                    score_text = data['data']['score']
                    if 'x' in score_text:
                        hit_number, hit_multiplier = parse_score_text(score_text)

                    reward = ai.train(
                        [target_x, target_y],
                        [landing_x, landing_y],
                        hit_number or 0,
                        hit_multiplier or 0
                    )
                    print(f"Training update completed (reward: {reward:.2f})")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in connection handler: {e}")
        import traceback
        traceback.print_exc()

async def main():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("Server started on ws://localhost:8765")
    print("\nOpen dartboard_feedback.html in your browser")
    print("Add ?mode=competing to URL for competing mode")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping AI server...")
