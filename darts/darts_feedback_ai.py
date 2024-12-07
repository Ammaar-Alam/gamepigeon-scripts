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
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

pyautogui.FAILSAFE = True

def parse_coords(coord_str):
    match = re.match(r'\((\d+),\s*(\d+)\)', coord_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_score_text(score_text):
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
        self.mode = mode
        print(f"AI initialized in {mode} mode")

        # Reference throw
        self.reference_throw = {
            'start_x': 1620,
            'start_y': 969,
            'end_x': 1620,
            'end_y': 469,
            'duration': 0.17
        }

        # Normalization constants
        # We'll scale x_adjust,y_adjust by 1/2000, landing_x, landing_y by 1/1000
        self.x_scale = 2000.0
        self.y_scale = 2000.0
        self.landing_scale = 1000.0

        # Data storage for supervised learning
        # Each entry: input: normalized (x_adj/x_scale, y_adj/x_scale), output: normalized (landing_x/landing_scale, landing_y/landing_scale)
        self.data_inputs = []
        self.data_outputs = []

        # Best throws dictionary
        self.best_throws = {}
        self.load_history()

        # Build model
        self.model = self.build_model()

        self.current_target = None
        self.current_throw_params = None
        self.last_score = 0.0

    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(2,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(2)  # Predict normalized landing_x, landing_y
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def save_history(self):
        with open('throw_history/best_throws.json', 'w') as f:
            json.dump(self.best_throws, f, indent=2)

    def load_history(self):
        try:
            with open('throw_history/best_throws.json', 'r') as f:
                self.best_throws = json.load(f)
        except FileNotFoundError:
            self.best_throws = {}

    def focus_iphone_window(self):
        script = """
        tell application "System Events"
            tell application process "iPhone Mirroring"
                set frontmost to true
            end tell
        end tell
        """
        subprocess.run(["osascript", "-e", script], capture_output=True)

    def move_iphone_mirror_to_corner(self):
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
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not move iPhone window: {e}")

    def normalize_input(self, x_adj, y_adj):
        return np.array([x_adj/self.x_scale, y_adj/self.y_scale], dtype=np.float32)

    def normalize_output(self, landing_x, landing_y):
        return np.array([landing_x/self.landing_scale, landing_y/self.landing_scale], dtype=np.float32)

    def denormalize_output(self, nx, ny):
        return nx * self.landing_scale, ny * self.landing_scale

    def predict_landing(self, x_adj, y_adj):
        if len(self.data_inputs) < 5:
            # Not enough data, just guess landing = reference center
            return 175.0, 100.0

        inp = self.normalize_input(x_adj, y_adj).reshape(1,-1)
        pred_norm = self.model.predict(inp, verbose=0)[0]
        landing_x, landing_y = self.denormalize_output(pred_norm[0], pred_norm[1])
        return landing_x, landing_y

    def retrain_model(self):
        if len(self.data_inputs) < 10:
            return
        X = np.array(self.data_inputs, dtype=np.float32)
        Y = np.array(self.data_outputs, dtype=np.float32)
        # Retrain model from scratch on all data for simplicity
        self.model = self.build_model()
        self.model.fit(X, Y, epochs=10, verbose=0)
        print("Forward model retrained on all collected data.")

    def store_throw_data(self, x_adj, y_adj, landing_x, landing_y, target_key, reward, throw_params):
        # Add to dataset
        self.data_inputs.append(self.normalize_input(x_adj, y_adj))
        self.data_outputs.append(self.normalize_output(landing_x, landing_y))

        # Update best throw if better
        if target_key not in self.best_throws or reward > self.best_throws[target_key].get('reward',0):
            self.best_throws[target_key] = {
                'params': throw_params.copy(),
                'reward': reward,
                'timestamp': datetime.now().isoformat()
            }
            print(f"New best throw for {target_key}! Reward: {reward:.2f}")

        self.save_history()
        # Retrain model
        self.retrain_model()

    def calculate_reward(self, target_coords, landing_coords, hit_number, hit_multiplier, target_number, target_multiplier, throw_failed=False):
        if throw_failed:
            return -0.5
        dx = target_coords[0]-landing_coords[0]
        dy = target_coords[1]-landing_coords[1]
        dist = math.sqrt(dx*dx+dy*dy)
        accuracy = math.exp(-dist/20.0)
        if hit_number > 0:
            number_val = hit_number / 20.0
            multiplier_val = hit_multiplier / 3.0
            quality = (number_val + multiplier_val)*0.5
            # If exact target
            t_mult = {'single':1,'double':2,'triple':3}[target_multiplier]
            if hit_number == target_number and hit_multiplier == t_mult:
                quality *= 2.0
            return (accuracy+quality)*0.5
        else:
            return accuracy*0.5

    def gradient_based_inverse(self, target_x, target_y, steps=30, lr=0.1):
        # If too few data points, just random guess
        if len(self.data_inputs) < 50:
            x_adj = random.uniform(-200,200)
            y_adj = random.uniform(-200,200)
            return x_adj, y_adj

        # Use TensorFlow gradients to solve inverse problem
        # We define a variable for (x_adj, y_adj) normalized
        x_var = tf.Variable(0.0, dtype=tf.float32)
        y_var = tf.Variable(0.0, dtype=tf.float32)

        # We'll do gradient descent to minimize error
        target_norm_x = target_x/self.landing_scale
        target_norm_y = target_y/self.landing_scale

        # We'll pass x_var,y_var through model and compute MSE
        for i in range(steps):
            with tf.GradientTape() as tape:
                inp = tf.stack([x_var, y_var])[None,:]
                pred = self.model(inp, training=False)[0]
                loss = tf.reduce_mean((pred[0]-target_norm_x)**2+(pred[1]-target_norm_y)**2)
            grads = tape.gradient(loss, [x_var,y_var])
            x_var.assign_sub(lr * grads[0])
            y_var.assign_sub(lr * grads[1])

        # Denormalize x_adj,y_adj
        x_adj = float(x_var.numpy()*self.x_scale)
        y_adj = float(y_var.numpy()*self.y_scale)
        return x_adj, y_adj

    def execute_throw(self, target_number, target_multiplier):
        self.current_target = (target_number, target_multiplier)
        target_key = f"{target_number}_{target_multiplier}"

        # We'll define some known coordinates for a few targets
        dartboard_coords = {
            '20_triple':[175,100],
            '3_triple':[175,250],
            '9_triple':[115,130],
            '2_triple':[220,235],
        }
        target_coords = dartboard_coords.get(target_key, [175,175])

        if self.mode == "competing":
            # Use best known or gradient-based inverse if not known
            if target_key in self.best_throws:
                throw_params = self.best_throws[target_key]['params'].copy()
                print(f"Using best throw for {target_key}")
            else:
                # Solve inverse using gradient-based approach
                x_adj, y_adj = self.gradient_based_inverse(target_coords[0], target_coords[1])
                end_x = int(self.reference_throw['end_x'] + x_adj)
                end_y = int(self.reference_throw['end_y'] - y_adj)
                throw_params = {
                    'start_x': self.reference_throw['start_x'],
                    'start_y': self.reference_throw['start_y'],
                    'end_x': end_x,
                    'end_y': end_y,
                    'duration': self.reference_throw['duration']
                }
                print(f"No best known for {target_key}, guessed via inverse.")
        else:
            # Training mode: Always use gradient-based inverse (once enough data), else random
            if len(self.data_inputs) < 50:
                # Not enough data to trust gradient approach, random exploration
                x_adj = random.uniform(-300,300)
                y_adj = random.uniform(-300,300)
            else:
                x_adj, y_adj = self.gradient_based_inverse(target_coords[0], target_coords[1])

            end_x = int(self.reference_throw['end_x'] + x_adj)
            end_y = int(self.reference_throw['end_y'] - y_adj)
            throw_params = {
                'start_x': self.reference_throw['start_x'],
                'start_y': self.reference_throw['start_y'],
                'end_x': end_x,
                'end_y': end_y,
                'duration': self.reference_throw['duration']
            }

        print(f"Executing throw with parameters: {throw_params}")
        self.focus_iphone_window()
        time.sleep(0.3)
        pyautogui.moveTo(throw_params['start_x'], throw_params['start_y'], duration=0.1)
        time.sleep(0.1)
        pyautogui.dragTo(
            throw_params['end_x'],
            throw_params['end_y'],
            duration=throw_params['duration'],
            button='left'
        )
        self.current_throw_params = throw_params
        return throw_params

    def process_landing(self, target_coords, landing_coords, hit_number, hit_multiplier, throw_failed):
        # Compute reward
        reward = self.calculate_reward(
            target_coords,
            landing_coords,
            hit_number,
            hit_multiplier,
            self.current_target[0],
            self.current_target[1],
            throw_failed=throw_failed
        )

        # Get offsets used
        x_adj = self.current_throw_params['end_x'] - self.reference_throw['end_x']
        y_adj = -(self.current_throw_params['end_y'] - self.reference_throw['end_y'])

        target_key = f"{self.current_target[0]}_{self.current_target[1]}"
        self.store_throw_data(x_adj, y_adj, landing_coords[0], landing_coords[1], target_key, reward, self.current_throw_params)
        self.last_score = reward
        return reward

async def handle_client(websocket, path):
    mode = "training"
    if "?" in path and "mode=competing" in path.split("?")[1]:
        mode = "competing"

    ai = DartsFeedbackAI(mode=mode)
    print(f"Client connected in {mode} mode")

    ai.focus_iphone_window()
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

            elif data['type'] == 'landing_data':
                target_x, target_y = parse_coords(data['data']['target'])
                landing_x, landing_y = parse_coords(data['data']['landing'])
                throw_failed = data['data'].get('throw_failed', False)

                if all([target_x, target_y, landing_x, landing_y]) and ai.current_target:
                    hit_number = 0
                    hit_multiplier = 0
                    score_text = data['data']['score']
                    if 'x' in score_text:
                        hit_number, hit_multiplier = parse_score_text(score_text)

                    reward = ai.process_landing(
                        [target_x, target_y],
                        [landing_x, landing_y],
                        hit_number or 0,
                        hit_multiplier or 0,
                        throw_failed
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
