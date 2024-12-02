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
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGMouseButtonLeft,
    kCGHIDEventTap,
    CGPointMake,
    CGEventSetIntegerValueField,
    kCGMouseEventPressure,
    kCGEventMouseMoved
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
    def __init__(self, debug_mode=False, mode="click"):
        # Add debug mode flag and interaction mode
        self.debug_mode = debug_mode
        self.mode = mode  # "click" or "move"

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

        # Get screen height for coordinate conversion
        self.screen_height = screen_height

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
        dictionary_path = "/Users/alam/Documents/GitHub/gamepigeon-scripts/wordhunt/dictionary.txt"
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
        
        # Calculate grid dimensions
        grid_width = int(width * .9) // 6  # 6 columns
        grid_height = bottom_crop - top_crop
        
        # Calculate board area within window
        board_top = int(self.window_height * 0.075)  # Adjust vertical position for anagrams
        board_space = self.window_height * 0.1  # Vertical spacing for single row
        board_width = self.window_width * 0.9  # Board takes up 80% of width
        board_x_offset = self.window_width * 0.05  # 10% margin on each side
        
        # Calculate letter positions
        self.letter_positions = []
        for i in range(6):
            # Calculate normalized position (0-1 range)
            norm_x = (i + 0.5) / 6  # 6 columns
            
            # Calculate actual screen coordinates
            screen_x = self.window_x + board_x_offset + (norm_x * board_width)
            y_pos = board_top +  board_space
            
            # Store the raw screen coordinates (will be converted to Quartz coords when used)
            self.letter_positions.append((screen_x, y_pos))
            
            # Draw debug visualization
            x_start = i * grid_width
            x_end = (i + 1) * grid_width
            cv2.rectangle(debug_img, 
                         (x_start, top_crop), 
                         (x_end, bottom_crop), 
                         (0, 255, 0), 2)
            cv2.circle(debug_img, 
                      (x_start + grid_width//2, top_crop + grid_height//2), 
                      2, (0, 0, 255), -1)
            cv2.putText(debug_img,
                       str(i + 1),
                       (x_start + grid_width//4, top_crop + grid_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 0, 0),
                       1)
        
        # Calculate ENTER button position (raw screen coordinates)
        enter_y = self.window_y - int(self.window_height * 0.075)  # Adjusted for anagrams
        enter_x = self.window_x + (self.window_width // 2)  # Center horizontally
        self.enter_position = (enter_x, enter_y)
        
        # Draw ENTER button bounds
        enter_width = width // 3
        enter_left = (width - enter_width) // 2
        enter_top = int(height * 0.35)
        enter_bottom = int(height * 0.45)
        cv2.rectangle(debug_img,
                     (enter_left, enter_top),
                     (enter_left + enter_width, enter_bottom),
                     (255, 0, 0), 2)
        cv2.circle(debug_img,
                  (width//2, (enter_top + enter_bottom)//2),
                  2, (255, 0, 0), -1)
        
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
        tile_width = img_width  // 6  # 6 tiles horizontally
        
        # Add spacing between tiles
        spacing = int(tile_width * 0.05)  # 10% of tile width for spacing
        effective_width = tile_width - spacing
        
        # Process each letter position
        for i in range(6):  # 6 tiles
            # Calculate tile bounds with spacing
            x_start = i * tile_width + spacing//2
            x_end = x_start + effective_width
            y_start = spacing//2
            y_end = img_height - spacing//2
            
            # Extract tile image
            tile_img = img[y_start:y_end, x_start:x_end]
            
            # Show tile for debugging
            cv2.imshow(f"Tile {i+1}", tile_img)
            cv2.waitKey(1)
            
            # Draw tile bounds on debug image
            cv2.rectangle(debug_img, 
                         (x_start, y_start), 
                         (x_end, y_end), 
                         (0, 255, 0), 2)
            
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
                
                # Draw letter and confidence on debug image
                cv2.putText(
                    debug_img,
                    f"{letter.upper()} ({confidence:.2f})",
                    (x_start + 5, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
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
        """Input a word by either clicking letters or just moving the mouse."""
        try:
            # Type each letter
            for letter in word:
                # Find position of letter
                for i, (x, y) in enumerate(self.letter_positions):
                    if self.letters[i] == letter:
                        # Convert to Quartz coordinates (flip y)
                        quartz_y = self.screen_height - y

                        if self.debug_mode:
                            print(f"{'Clicking' if self.mode == 'click' else 'Moving to'} letter {letter.upper()} at ({x}, {quartz_y})")

                        # Move to letter position
                        move_event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, CGPointMake(x, quartz_y), kCGMouseButtonLeft)
                        CGEventPost(kCGHIDEventTap, move_event)
                        time.sleep(0.1)

                        # Click letter if in click mode
                        if self.mode == "click":
                            down_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, CGPointMake(x, quartz_y), kCGMouseButtonLeft)
                            up_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, CGPointMake(x, quartz_y), kCGMouseButtonLeft)

                            CGEventPost(kCGHIDEventTap, down_event)
                            # time.sleep(0.1)
                            CGEventPost(kCGHIDEventTap, up_event)
                        break

            # Move to or click ENTER
            if self.enter_position:
                x, y = self.enter_position
                # Convert to Quartz coordinates (flip y)
                quartz_y = self.screen_height - y

                if self.debug_mode:
                    print(f"{'Clicking' if self.mode == 'click' else 'Moving to'} ENTER at ({x}, {quartz_y})")

                # Move to ENTER button
                move_event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, CGPointMake(x, quartz_y), kCGMouseButtonLeft)
                CGEventPost(kCGHIDEventTap, move_event)
                # time.sleep(0.1)

                # Click ENTER if in click mode
                if self.mode == "click":
                    down_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, CGPointMake(x, quartz_y), kCGMouseButtonLeft)
                    up_event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, CGPointMake(x, quartz_y), kCGMouseButtonLeft)

                    CGEventPost(kCGHIDEventTap, down_event)
                    # time.sleep(0.1)
                    CGEventPost(kCGHIDEventTap, up_event)
                    # time.sleep(0.3)

        except Exception as e:
            print(f"Error inputting word {word}: {e}")

    def solve(self):
        """Main solving loop."""
        # Start timer for game time limit
        start_time = time.time()
        round_duration = 61.5  # 1:00 + 5 seconds buffer

        # Focus window before capture
        focus_iphone_window()
        time.sleep(0.5)

        # Capture and process the board
        print("\nCapturing board...")
        board_img = self.capture_board()

        print("\nProcessing image...")
        letters = self.process_image(board_img)

        print("\nDetected letters:", " ".join(letter.upper() for letter in letters))

        # Add delay to keep windows open and wait for user input
        print("\nPress Enter to continue or Q to quit...")
        # try:
        #     while True:
        #         if keyboard.is_pressed('q'):
        #             print("\nQuitting...")
        #             return
        #         if keyboard.is_pressed('enter'):
        #             break
        #         time.sleep(0.1)
        # except Exception as e:
        #     print(f"Error waiting for input: {e}")
        #     return

        # Find all possible words
        words = self.find_words(letters)
        print(f"\nFound {len(words)} possible words")

        if len(words) == 0:
            print("\nNo words found. Press Enter to quit...")
            return

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

            print(f"{'Inputting' if self.mode == 'click' else 'Showing'} word: {word}")
            self.input_word(word)
            # time.sleep(0.2)

    def __del__(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        cv2.destroyAllWindows()

def test_bounds():
    """Move mouse around the border of the game area to visualize boundaries."""
    try:
        # Initialize window position
        window_info = find_iphone_mirror_window()
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")

        window_x, window_y, window_width, window_height = window_info
        screen_height = pyautogui.size()[1]

        # Calculate board area for Anagrams (adjusted percentages)
        board_top = int(window_height * 0.075)  # Letters are near top for Anagrams 
        board_space = window_height * 0.1  # Vertical spacing for single row
        board_width = window_width * 0.9  # Board takes up 90% of width
        board_x_offset = window_width * 0.05  # 5% margin on each side

        # Calculate corner points (clockwise from top-left)
        corners = [
            # Top left
            (window_x + board_x_offset, screen_height - (window_y + board_top)),
            # Top right 
            (window_x + board_x_offset + board_width, screen_height - (window_y + board_top)),
            # Bottom right
            (window_x + board_x_offset + board_width, screen_height - (window_y + board_top + board_space)),
            # Bottom left
            (window_x + board_x_offset, screen_height - (window_y + board_top + board_space))
        ]

        print("\nMoving mouse to show game boundaries...")
        print("Press Ctrl+C to stop")

        # Move to each corner and pause
        while True:  # Loop until interrupted
            for i, (x, y) in enumerate(corners):
                source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
                point = CGPointMake(x, y)
                move_event = CGEventCreateMouseEvent(source, kCGEventMouseMoved, point, kCGMouseButtonLeft)
                CGEventPost(kCGHIDEventTap, move_event)
                print(f"Corner {i+1}: ({x:.1f}, {y:.1f})")
                time.sleep(1)  # Pause at each corner

    except KeyboardInterrupt:
        print("\nBoundary test stopped")
    except Exception as e:
        print(f"Error during boundary test: {e}")

def test_letter_positions():
    """Move mouse to each letter position and ENTER button to visualize all positions."""
    try:
        # Initialize window position
        window_info = find_iphone_mirror_window()
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")

        window_x, window_y, window_width, window_height = window_info
        screen_height = pyautogui.size()[1]

        # Calculate board area for Anagrams (adjusted percentages)
        board_top = int(window_height * 0.075)  # Letters are near top for Anagrams
        board_space = window_height * 0.1  # Vertical spacing for single row
        board_width = window_width * 0.9  # Board takes up 90% of width
        board_x_offset = window_width * 0.05  # 5% margin on each side

        # Calculate ENTER button position
        enter_y = window_y + int(window_height * 0.925)  # Near bottom of screen
        enter_x = window_x + (window_width // 2)  # Center horizontally

        print("\nMoving mouse to show letter positions and ENTER button...")
        print("Press Ctrl+C to stop")

        while True:  # Loop until interrupted
            # Loop through each letter position (6 letters)
            for i in range(6):
                # Calculate normalized position (0-1 range)
                norm_x = (i + 0.5) / 6  # 6 columns

                # Calculate actual screen coordinates
                screen_x = window_x + board_x_offset + (norm_x * board_width)
                y_pos = window_y + board_top + board_space
                quartz_y = screen_height - y_pos

                # Move to position
                point = CGPointMake(screen_x, quartz_y)
                source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
                move_event = CGEventCreateMouseEvent(source, kCGEventMouseMoved, point, kCGMouseButtonLeft)
                CGEventPost(kCGHIDEventTap, move_event)

                print(f"Letter position {i+1}: ({screen_x:.1f}, {quartz_y:.1f})")
                time.sleep(1)  # Pause at each position

            # Move to ENTER button
            quartz_enter_y = screen_height - enter_y
            point = CGPointMake(enter_x, quartz_enter_y)
            source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            move_event = CGEventCreateMouseEvent(source, kCGEventMouseMoved, point, kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, move_event)
            print(f"ENTER button: ({enter_x:.1f}, {quartz_enter_y:.1f})")
            time.sleep(1)

            print("\nCompleted one cycle, starting over...")
            time.sleep(1)  # Pause between cycles

    except KeyboardInterrupt:
        print("\nLetter position test stopped")
    except Exception as e:
        print(f"Error during letter position test: {e}")

def main():
    print("\nAnagrams Solver")
    print("1. Open the Anagrams game")
    print("2. Press Enter to position the window")
    print("3. Choose mode:")
    print("   S - Solve (detect and input words)")
    print("   B - Show boundaries (4 corners of letter grid)")
    print("   L - Show letter positions (one at a time) and ENTER button")
    print("   Q - Quit")
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

        while True:
            choice = input("\nEnter choice (S/B/L/Q): ").lower()
            if choice == 's':
                solver.solve()
            elif choice == 'b':
                test_bounds()
            elif choice == 'l':
                test_letter_positions()
            elif choice == '':
                solver.solve()
            elif choice =='q':
                print("Quitting...")
                break
            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\nSolver stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
