import mss
import numpy as np
import cv2
import pyautogui
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import List, Tuple, Set, Dict
import os
import subprocess
import keyboard
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
    kCGHIDEventTap,
    CGPointMake
)

# Import window management functions
from move_window import find_iphone_mirror_window, move_iphone_mirror_to_corner, focus_iphone_window

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word

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

class AnagramsSolver:
    def __init__(self):
        # Initialize window position
        window_info = find_iphone_mirror_window()
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")
        
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        print(f"Screen dimensions: {screen_width}x{screen_height}")
        
        # Store window info
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
        
        # Initialize CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LetterCNN().to(self.device)
        
        # Load trained weights (using the same model as WordHunt)
        try:
            self.model.load_state_dict(torch.load('/Users/alam/Documents/GitHub/gamepigeon-scripts/wordhunt/letter_recognition_model.pth'))
            self.model.eval()
            print("Loaded trained model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise Exception("Could not load trained model!")
        
        # Letter mapping
        self.idx_to_letter = {i: chr(i + 65).lower() for i in range(26)}
        
        # Create debug window
        cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug View", self.window_width, self.window_height)
        
        # Load dictionary and build trie
        self.dictionary, self.trie = self.load_dictionary()
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True
        
        # Save original window_y for screen capture
        self.window_y_orig = self.window_y
        
        # Adjust window_y to match Quartz coordinate system
        self.window_y = screen_height - (self.window_y + self.window_height)
        print(f"Adjusted window_y: {self.window_y}")
        
        # Add a stop flag
        self.stop_flag = False
        
        # Initialize keyboard listener
        keyboard.on_press(self.on_press)
        
        # Store letter positions and ENTER position
        self.letter_positions = []
        self.enter_position = None
        
        # Store detected letters
        self.letters = []

    def on_press(self, key):
        """Handle keyboard events."""
        try:
            if key.char.lower() == 'q':
                print("\nStopping solver...")
                self.stop_flag = True
        except AttributeError:
            pass

    def load_dictionary(self) -> Tuple[Set[str], Trie]:
        """Load dictionary and build trie."""
        words = set()
        trie = Trie()
        dictionary_path = "../wordhunt/dictionary.txt"
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if len(word) >= 3:  # Only words of 3+ letters
                        words.add(word)
                        trie.insert(word)
        return words, trie

    def detect_game_board(self, img: np.ndarray) -> Tuple[dict, np.ndarray]:
        """Detect the game board and calculate bounds."""
        debug_img = img.copy()
        
        # Crop to the region where the letters are likely to be
        height, width = img.shape[:2]
        top_crop = int(height * 0.78)  # Letters are near bottom
        bottom_crop = int(height * 0.86)
        
        cropped = img[top_crop:bottom_crop, :]
        cv2.imshow("Cropped Input", cropped)
        cv2.waitKey(1)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # Show HSV channels separately for debugging
        h, s, v = cv2.split(hsv)
        cv2.imshow("HSV - H Channel", h)
        cv2.imshow("HSV - S Channel", s)
        cv2.imshow("HSV - V Channel", v)
        cv2.waitKey(1)
        
        # Sample the center of a letter tile for color calibration
        center_y = cropped.shape[0] // 2
        center_x = cropped.shape[1] // 2
        sample_color = hsv[center_y, center_x]
        print(f"Sample HSV color at center: {sample_color}")
        
        # Define beige color range for letter tiles (adjusted based on sample)
        lower_beige = np.array([15, 0, 200])  # Very light beige
        upper_beige = np.array([30, 50, 255])  # Allow higher brightness
        beige_mask = cv2.inRange(hsv, lower_beige, upper_beige)
        
        # Show the initial mask
        cv2.imshow("Initial Mask", beige_mask)
        cv2.waitKey(1)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel
        beige_mask = cv2.morphologyEx(beige_mask, cv2.MORPH_CLOSE, kernel)
        beige_mask = cv2.morphologyEx(beige_mask, cv2.MORPH_OPEN, kernel)
        
        # Show the processed mask
        cv2.imshow("Processed Mask", beige_mask)
        cv2.waitKey(1)
        
        # Find contours
        contours, _ = cv2.findContours(beige_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all contours for debugging
        contour_debug = cropped.copy()
        cv2.drawContours(contour_debug, contours, -1, (0,255,0), 2)
        cv2.imshow("All Contours", contour_debug)
        cv2.waitKey(1)
        
        # Filter and sort contours
        valid_contours = []
        min_area = 500  # Increased minimum area
        max_area = 5000  # Maximum area threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.8 < aspect_ratio < 1.2:  # Square-ish tiles
                    valid_contours.append(contour)
                    # Print debug info
                    print(f"Valid contour - Area: {area}, Aspect ratio: {aspect_ratio:.2f}, Position: ({x}, {y})")
        
        print(f"Found {len(valid_contours)} valid contours")
        
        # Sort left to right
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Draw valid contours
        valid_debug = cropped.copy()
        cv2.drawContours(valid_debug, valid_contours, -1, (0,0,255), 2)
        cv2.imshow("Valid Contours", valid_debug)
        cv2.waitKey(1)
        
        # Store letter positions
        self.letter_positions = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = self.window_x + x + (w / 2)
            center_y = self.window_y_orig + top_crop + y + (h / 2)
            self.letter_positions.append((center_x, center_y))
            
            # Draw bounding box and center
            cv2.rectangle(debug_img, (x, y + top_crop), (x + w, y + h + top_crop), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(x + w/2), int(y + h/2 + top_crop)), 2, (0, 0, 255), -1)
            
            # Draw tile number for debugging
            cv2.putText(
                debug_img,
                str(len(self.letter_positions)),
                (int(x + w/2), int(y + h/2 + top_crop) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
        
        # Detect ENTER button
        enter_top = int(height * 0.50)
        enter_bottom = int(height * 0.60)
        enter_area = img[enter_top:enter_bottom, :]
        
        # Convert to HSV
        enter_hsv = cv2.cvtColor(enter_area, cv2.COLOR_BGR2HSV)
        
        # Define gray/purple color range for ENTER button
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 30, 255])
        gray_mask = cv2.inRange(enter_hsv, lower_gray, upper_gray)
        
        cv2.imshow("ENTER Button Mask", gray_mask)
        cv2.waitKey(1)
        
        # Find ENTER button contour
        enter_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if enter_contours:
            # Find largest contour near center
            center_x = width / 2
            closest_contour = None
            min_distance = float('inf')
            
            for contour in enter_contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Minimum area for ENTER button
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                contour_center = x + (w / 2)
                distance = abs(contour_center - center_x)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour
            
            if closest_contour is not None:
                x, y, w, h = cv2.boundingRect(closest_contour)
                self.enter_position = (
                    self.window_x + x + (w / 2),
                    self.window_y_orig + enter_top + (h / 2)
                )
                cv2.rectangle(debug_img, (x, y + enter_top), (x + w, y + h + enter_top), (255, 0, 0), 2)
                cv2.circle(debug_img, (int(x + w/2), int(y + h/2 + enter_top)), 2, (255, 0, 0), -1)
        
        cv2.imshow("Debug View", debug_img)
        cv2.waitKey(1)
        
        return {
            'top': top_crop,
            'bottom': bottom_crop,
            'letter_positions': self.letter_positions,
            'enter_position': self.enter_position
        }, cropped

    def capture_board(self) -> np.ndarray:
        """Capture the game board."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        
        cv2.imshow("Full Capture", img)
        cv2.waitKey(1)
        
        _, board_img = self.detect_game_board(img)
        
        if board_img is not None and board_img.size > 0:
            cv2.imshow("Board Capture", board_img)
            cv2.waitKey(1)
        else:
            print("Error: Board capture is empty!")
        
        return board_img

    def process_image(self, img: np.ndarray) -> List[str]:
        """Process the captured image to extract letters."""
        if img is None or img.size == 0:
            print("Error: Empty image provided to process_image")
            return []
        
        letters = []
        debug_img = img.copy()
        
        # Define transforms for CNN
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Calculate tile size based on image dimensions
        img_height, img_width = img.shape[:2]
        tile_width = img_width // 6  # 6 tiles horizontally
        tile_height = img_height     # Full height for each tile
        
        # Process each letter position
        for i in range(6):  # 6 tiles
            # Calculate tile bounds
            x_start = i * tile_width
            x_end = (i + 1) * tile_width
            y_start = 0
            y_end = tile_height
            
            # Add margin to focus on letter
            margin_x = int(tile_width * 0.2)
            margin_y = int(tile_height * 0.2)
            x_start += margin_x
            x_end -= margin_x
            y_start += margin_y
            y_end -= margin_y
            
            # Extract tile image
            tile_img = img[y_start:y_end, x_start:x_end]
            
            # Show tile for debugging
            cv2.imshow(f"Tile {i+1}", tile_img)
            cv2.waitKey(1)
            
            try:
                # Preprocess and get prediction
                tensor = transform(tile_img)
                tensor = tensor.unsqueeze(0)
                tensor = tensor.to(self.device)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    pred_idx = output.argmax(dim=1).item()
                    confidence = torch.exp(output.max()).item()
                    
                    letter = self.idx_to_letter[pred_idx]
                    
                    if confidence > 0.5:  # Confidence threshold
                        letters.append(letter)
                        print(f"Detected letter {i}: {letter.upper()} (confidence: {confidence:.2f})")
                    else:
                        print(f"Low confidence detection for letter {i}: {letter.upper()} ({confidence:.2f})")
                
                # Draw debug info
                cv2.putText(
                    debug_img,
                    letter.upper(),
                    (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                # Draw tile bounds
                cv2.rectangle(
                    debug_img,
                    (x_start, y_start),
                    (x_end, y_end),
                    (0, 255, 0),
                    2
                )
                
            except Exception as e:
                print(f"Error processing letter {i}: {e}")
                continue
        
        cv2.imshow("Debug View", debug_img)
        cv2.waitKey(1)
        
        self.letters = letters
        return letters

    def find_words(self, letters: List[str]) -> List[str]:
        """Find all possible words that can be made from the given letters."""
        def find_words_recursive(node: TrieNode, remaining_letters: List[str], 
                               current_word: str, words: Set[str]):
            if node.is_word and len(current_word) >= 3:
                words.add(current_word)
            
            # Try each remaining letter
            for i, letter in enumerate(remaining_letters):
                if letter in node.children:
                    # Remove the used letter and recurse
                    new_remaining = remaining_letters[:i] + remaining_letters[i+1:]
                    find_words_recursive(
                        node.children[letter],
                        new_remaining,
                        current_word + letter,
                        words
                    )
        
        found_words = set()
        find_words_recursive(self.trie.root, letters, "", found_words)
        
        # Sort by length (longest first) then alphabetically
        return sorted(list(found_words), key=lambda x: (-len(x), x))

    def input_word(self, word: str):
        """Input a word by tapping letters and pressing ENTER."""
        try:
            screen_height = pyautogui.size()[1]
            
            # Type each letter
            for letter in word:
                # Find position of letter
                for i, (x, y) in enumerate(self.letter_positions):
                    if self.letters[i] == letter:
                        # Convert to Quartz coordinates
                        quartz_y = screen_height - y
                        
                        # Click letter
                        point = CGPointMake(x, quartz_y)
                        down_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, kCGMouseButtonLeft)
                        up_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, kCGMouseButtonLeft)
                        
                        CGEventPost(kCGHIDEventTap, down_event)
                        time.sleep(0.1)
                        CGEventPost(kCGHIDEventTap, up_event)
                        time.sleep(0.15)
                        break
            
            # Add slight delay before pressing ENTER
            time.sleep(0.2)
            
            # Press ENTER
            if self.enter_position:
                x, y = self.enter_position
                quartz_y = screen_height - y
                
                point = CGPointMake(x, quartz_y)
                down_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, kCGMouseButtonLeft)
                up_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, kCGMouseButtonLeft)
                
                CGEventPost(kCGHIDEventTap, down_event)
                time.sleep(0.1)
                CGEventPost(kCGHIDEventTap, up_event)
                time.sleep(0.3)
            
        except Exception as e:
            print(f"Error inputting word {word}: {e}")

    def solve(self):
        """Main solving loop."""
        # Start timer for game time limit
        start_time = time.time()
        round_duration = 65  # 1:00 + 5 seconds buffer
        
        # Focus window before capture
        focus_iphone_window()
        time.sleep(0.5)
        
        # Capture and process the board
        print("\nCapturing board...")
        board_img = self.capture_board()
        
        print("\nProcessing image...")
        letters = self.process_image(board_img)
        
        print("\nDetected letters:", " ".join(letter.upper() for letter in letters))
        
        # Find all possible words
        words = self.find_words(letters)
        print(f"\nFound {len(words)} possible words")
        
        # Sort words by length (longest first) then alphabetically
        words.sort(key=lambda x: (-len(x), x))
        
        # Make sure window is still focused
        focus_iphone_window()
        time.sleep(0.5)
        
        # Input each word
        for word in words:
            # Check time limit and stop flag
            elapsed_time = time.time() - start_time
            if elapsed_time >= round_duration:
                print("\nTime limit reached! Stopping solver...")
                break
            
            if self.stop_flag:
                print("Solver stopped by user (Q pressed)")
                break
            
            print(f"Inputting word: {word}")
            self.input_word(word)
            time.sleep(0.2)

    def __del__(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        cv2.destroyAllWindows()

def main():
    print("\nAnagrams Solver")
    print("1. Open the Anagrams game")
    print("2. Press Enter to position the window")
    print("3. Press Enter again to start solving")
    print("   Q - Quit during solving")
    input()
    
    try:
        # Move window to corner
        print("\nMoving iPhone window to corner...")
        window_info = move_iphone_mirror_to_corner()
        if not window_info:
            print("Failed to position window!")
            return
        
        time.sleep(1)
        
        solver = AnagramsSolver()
        input("\nPress Enter to start solving...")
        solver.solve()
        
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
