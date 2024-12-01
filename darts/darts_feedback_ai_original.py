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
            'duration': 0.135  # Faster duration for better response
        }

        # Load throw history first
        self.throw_history = {}
        self.best_throws = {}  # Store best throws per target
        self.load_history()

        # Initialize state
        self.last_throw = None
        self.current_target = None
        self.last_target_coords = None
        self.last_adjustments = {'x': 0.0, 'y': 0.0}
        self.throw_params = None

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

    def load_history(self):
        """Load throw history and identify best throws."""
        try:
            with open('throw_history/throws.json', 'r') as f:
                self.throw_history = json.load(f)
                
            # Identify best throws for each target
            for target_key, throws in self.throw_history.items():
                if throws:
                    # Parse target info
                    target_number = int(target_key.split('_')[0])
                    target_multiplier = self.multiplier_map[target_key.split('_')[1]]
                    
                    # Find throws that hit the exact target with high accuracy
                    exact_hits = []
                    for throw in throws:
                        # Try to get hit info from score text if not in throw data
                        if 'hit_number' not in throw and 'score' in throw:
                            score_text = throw.get('score', '')
                            if isinstance(score_text, str) and 'x' in score_text:
                                hit_number, hit_multiplier = parse_score_text(score_text)
                                if hit_number is not None:
                                    throw['hit_number'] = hit_number
                                    throw['hit_multiplier'] = hit_multiplier
                        
                        # Check if it's an exact hit
                        if (throw.get('hit_number', 0) == target_number and 
                            throw.get('hit_multiplier', 0) == target_multiplier and
                            throw.get('score', 0) > 0.7):
                            exact_hits.append(throw)
                    
                    if exact_hits:
                        # Get the most accurate exact hit
                        best_throw = max(exact_hits, key=lambda t: t.get('score', 0))
                        # Round coordinates to integers for consistency
                        best_throw['x_adjust'] = round(best_throw['x_adjust'])
                        best_throw['y_adjust'] = round(best_throw['y_adjust'])
                        self.best_throws[target_key] = best_throw
                        print(f"Loaded best throw for {target_key} with score {min(best_throw.get('score', 0), 1.0):.2f}")

        except FileNotFoundError:
            self.throw_history = {}

    def save_history(self):
        """Save throw history to file."""
        print("Saving throw history...")
        with open('throw_history/throws.json', 'w') as f:
            json.dump(self.throw_history, f, indent=2)
        print("Throw history saved")

    def get_target_key(self, number, multiplier):
        """Convert number and multiplier to consistent target key format."""
        return f"{number}_{multiplier}"

    def predict_adjustments(self, target_number, target_multiplier, target_coords):
        """Predict throw adjustments based on target position."""
        target_key = self.get_target_key(target_number, target_multiplier)

        # In competing mode, strictly use best known throw if available
        if self.mode == "competing" and target_key in self.best_throws:
            best_throw = self.best_throws[target_key]
            print(f"Using best known throw for {target_key} with score {min(best_throw.get('score', 0), 1.0):.2f}")
            return best_throw.get('x_adjust', 0), best_throw.get('y_adjust', 0)

        # Training mode logic
        # Calculate base adjustments from board coordinates to screen coordinates
        x_diff = target_coords[0] - 175  # Difference from board center x
        y_diff = target_coords[1] - 175  # Difference from board center y

        # Scale the differences to screen coordinates
        x_adjust = x_diff * self.x_scale
        y_adjust = y_diff * self.y_scale

        print(f"Initial adjustments - x: {x_adjust:.1f}, y: {y_adjust:.1f}")

        # If we have landing data from last throw, apply corrections
        if self.last_throw and self.last_adjustments:
            aimed_coords_x = 175 + (self.last_adjustments['x'] / self.x_scale)
            aimed_coords_y = 175 + (self.last_adjustments['y'] / self.y_scale)

            error_x = self.last_throw[0] - aimed_coords_x
            error_y = self.last_throw[1] - aimed_coords_y

            # Apply scaled corrections
            x_adjust -= error_x * self.x_scale * 0.5
            y_adjust -= error_y * self.y_scale * 0.5

            print(f"Error adjustments - x: {-error_x * self.x_scale * 0.5:.1f}, y: {-error_y * self.y_scale * 0.5:.1f}")

        # Print debugging info
        print(f"Target coordinates: {target_coords}")
        print(f"Final adjustments: ({x_adjust:.1f}, {y_adjust:.1f})")

        # Clamp adjustments
        max_adjust = self.MAX_ADJUST * 0.8  # 160
        x_adjust = np.clip(x_adjust, -max_adjust, max_adjust)
        y_adjust = np.clip(y_adjust, -max_adjust, max_adjust)

        # Round adjustments to integers for consistency
        x_adjust = round(x_adjust)
        y_adjust = round(y_adjust)

        return x_adjust, y_adjust

    def train(self, target_number, target_multiplier, hit_number, hit_multiplier, throw_params, target_coords, landing_coords):
        """Train on throw result with improved scoring."""
        # Calculate distance error
        error_x = target_coords[0] - landing_coords[0]
        error_y = target_coords[1] - landing_coords[1]
        distance_error = math.sqrt(error_x**2 + error_y**2)

        # Calculate base score based on distance (exponential falloff)
        base_score = math.exp(-distance_error / 50)
        # Cap base score at 1.0
        base_score = min(base_score, 1.0)

        # Apply bonus if we hit the exact target
        score = base_score
        if hit_number == target_number and hit_multiplier == self.multiplier_map[target_multiplier]:
            score = min(base_score * 1.5, 1.0)  # Apply bonus but cap at 1.0
            print(f"Bonus applied for hitting exact target! New score: {score:.2f}")

        # Store throw data
        target_key = self.get_target_key(target_number, target_multiplier)
        if target_key not in self.throw_history:
            self.throw_history[target_key] = []

        # Calculate adjustments made for this throw
        x_adjust = round(throw_params['end_x'] - self.reference_throw['end_x'])
        y_adjust = round(throw_params['end_y'] - self.reference_throw['end_y'])

        throw_data = {
            'x_adjust': x_adjust,
            'y_adjust': y_adjust,
            'target_coords': target_coords,
            'landing_coords': landing_coords,
            'score': score,
            'distance_error': distance_error,
            'hit_number': hit_number,
            'hit_multiplier': hit_multiplier,
            'duration': throw_params['duration'],  # Store duration for exact reproduction
            'timestamp': datetime.now().isoformat()
        }

        self.throw_history[target_key].append(throw_data)
        print(f"Added throw data for {target_key}: {throw_data}")
        print(f"Adjustments made: x={x_adjust:.1f}, y={y_adjust:.1f}")
        print(f"Distance error: {distance_error:.1f} pixels")

        # Update best throw if this was better and hit the exact target
        if hit_number == target_number and hit_multiplier == self.multiplier_map[target_multiplier]:
            if target_key not in self.best_throws or score > self.best_throws[target_key].get('score', 0):
                if score > 0.7:
                    self.best_throws[target_key] = throw_data
                    print(f"New best throw recorded for {target_key}!")

        # Keep only last 10 throws for each target, sorted by score
        if len(self.throw_history[target_key]) > 10:
            self.throw_history[target_key] = sorted(
                self.throw_history[target_key],
                key=lambda t: t.get('score', 0),
                reverse=True
            )[:10]

        self.save_history()
        self.last_throw = landing_coords
        self.last_target_coords = target_coords  # Update after training

        return score

    def execute_throw(self, target_number, target_multiplier):
        """Execute throw with predicted adjustments."""
        self.current_target = (target_number, target_multiplier)

        # Get target coordinates
        target_key = self.get_target_key(target_number, target_multiplier)
        target_coords = None

        # Dartboard UI coordinates
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

        # In competing mode, use best throw if available
        if self.mode == "competing" and target_key in self.best_throws:
            best_throw = self.best_throws[target_key]
            # Use exact integer values from best throw
            target_x = int(self.reference_throw['end_x'] + best_throw['x_adjust'])
            target_y = int(self.reference_throw['end_y'] + best_throw['y_adjust'])
            duration = best_throw['duration']  # Use exact duration from best throw
            print(f"Using best throw coordinates: ({target_x}, {target_y}) with duration {duration}")
        else:
            # Training mode logic
            # Prioritize dartboard UI coordinates
            if target_key in dartboard_coords:
                target_coords = dartboard_coords[target_key]
                print(f"Using dartboard UI coordinates for {target_key}: {target_coords}")
            elif target_key in self.throw_history and len(self.throw_history[target_key]) > 0:
                best_throw = max(self.throw_history[target_key], key=lambda t: t.get('score', 0))
                if best_throw.get('score', 0) > 0.3:
                    target_coords = best_throw.get('target_coords')
                    if target_coords:
                        target_coords = tuple(target_coords)
                        print(f"Using historical target coordinates for {target_key}: {target_coords}")

            if not target_coords:
                target_coords = [175, 175]
                print(f"No coordinates found for {target_key}, using default: {target_coords}")

            # Get adjustments
            x_adjust, y_adjust = self.predict_adjustments(
                target_number,
                target_multiplier,
                target_coords
            )

            # Store the adjustments for error correction
            self.last_adjustments = {'x': x_adjust, 'y': y_adjust}

            # Calculate target position using integer values
            target_x = int(self.reference_throw['end_x'] + x_adjust)
            target_y = int(self.reference_throw['end_y'] + y_adjust)

            # Use fixed duration for consistency
            duration = 0.17

        # Ensure target positions are within screen boundaries
        target_x = int(np.clip(target_x, 0, self.screen_width))
        target_y = int(np.clip(target_y, 0, self.screen_height))

        print(f"Calculated duration: {duration:.3f} seconds")
        print(f"Target coordinates: ({target_x}, {target_y})")

        # Execute the throw
        try:
            self.focus_iphone_window()
        except Exception as e:
            print(f"Warning: Could not focus iPhone window: {e}")
            
        time.sleep(0.5)

        print("Moving to start position...")
        pyautogui.moveTo(
            self.reference_throw['start_x'],
            self.reference_throw['start_y'],
            duration=0.1
        )
        time.sleep(0.1)

        print("Performing throw...")
        pyautogui.dragTo(
            target_x,
            target_y,
            duration=duration,
            tween=pyautogui.easeOutQuad,
            button='left'
        )

        self.throw_params = {
            'start_x': self.reference_throw['start_x'],
            'start_y': self.reference_throw['start_y'],
            'end_x': target_x,
            'end_y': target_y,
            'duration': duration
        }

        return self.throw_params

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
            if result.stderr:
                print(f"Warning: Could not move iPhone window: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not move iPhone window: {e}")

async def handle_client(websocket, path):
    """Handle WebSocket connection from feedback UI."""
    # Get mode from query parameter
    mode = "training"  # default mode
    if "?" in path:
        query = path.split("?")[1]
        if "mode=competing" in query:
            mode = "competing"
    
    ai = DartsFeedbackAI(mode=mode)
    print(f"Client connected in {mode} mode")

    # Move window to corner
    print("\nPositioning iPhone window...")
    ai.move_iphone_mirror_to_corner()

    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received message: {data}")

            if data['type'] == 'throw_request':
                # Execute throw
                target = data['data']
                print(f"\nAiming for {target['multiplier']} {target['number']}")

                throw_params = ai.execute_throw(
                    int(target['number']),
                    target['multiplier']
                )
                print(f"Threw to: ({throw_params['end_x']}, {throw_params['end_y']})")

            elif data['type'] == 'landing_data':
                # Parse coordinates regardless of score
                target_x, target_y = parse_coords(data['data']['target'])
                landing_x, landing_y = parse_coords(data['data']['landing'])

                if target_x is not None and landing_x is not None and ai.current_target is not None:
                    target_number, target_multiplier = ai.current_target

                    # Get hit info if available, otherwise use 0
                    hit_number = 0
                    hit_multiplier = 0
                    score_text = data['data']['score']
                    if 'x' in score_text:
                        hit_number = int(score_text.split('x')[0].strip())
                        hit_multiplier = int(score_text.split('=')[0].split('x')[1].strip())

                    # Train on result
                    score = ai.train(
                        target_number,
                        target_multiplier,
                        hit_number,
                        hit_multiplier,
                        ai.throw_params,
                        [target_x, target_y],
                        [landing_x, landing_y]
                    )

                    print(f"AI updated based on feedback (score: {score:.2f})")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in connection handler: {e}")

async def main():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("Server started on ws://localhost:8765")
    print("\nOpen dartboard_feedback.html in your browser")
    print("Add ?mode=competing to URL for competing mode")
    print("Then use the controls to execute throws and provide feedback")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping AI server...")