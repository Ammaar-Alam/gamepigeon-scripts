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
import pytesseract
from typing import List, Tuple, Set
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

# Import additional required Quartz modules
import Quartz

def create_mouse_event(type, point, source=None):
    """Create a mouse event with proper source state and pressure."""
    if source is None:
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    event = CGEventCreateMouseEvent(source, type, point, kCGMouseButtonLeft)
    # Add pressure for more reliable mouse events
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
            
            if 'iPhone Mirroring' in str(owner):  # Updated to match exact window name
                bounds = window.get(Quartz.kCGWindowBounds)
                if bounds:
                    print(f"Found iPhone Mirroring window: {bounds}")
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
        tell process "iPhone Mirroring"
            set frontmost to true
            delay 0.5
            click menu item "Bottom Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1
        end tell
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
        tell process "iPhone Mirroring"
            set frontmost to true
        end tell
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.1)
    except Exception as e:
        print(f"Focus failed: {e}")

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

def preprocess_for_cnn(cell_img: np.ndarray) -> torch.Tensor:
    """Preprocess image for CNN input."""
    # Convert to grayscale if needed
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    
    # Resize to fixed size
    resized = cv2.resize(gray, (28, 28))
    
    # Convert to PIL Image
    pil_img = Image.fromarray(resized)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Apply transforms
    tensor = transform(pil_img)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor

def train_model(model, train_loader, device):
    """Training function for the CNN."""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):  # Number of epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

def load_model(path):
    """Load a trained model."""
    model = LetterCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def create_mouse_event(type, point, source=None):
    """Create a mouse event with proper source state."""
    if source is None:
        source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    return CGEventCreateMouseEvent(source, type, point, kCGMouseButtonLeft)

class WordHuntSolver:
    def __init__(self):
        # Initialize window position using the accurate window finder
        window_info = find_iphone_mirror_window()
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")
        
        # Get screen dimensions for coordinate conversion
        screen_width, screen_height = pyautogui.size()
        print(f"Screen dimensions: {screen_width}x{screen_height}")
        
        # Store original window info
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
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load('letter_recognition_model.pth'))
            self.model.eval()
            print("Loaded trained model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise Exception("Could not load trained model!")
        
        # Letter mapping
        self.idx_to_letter = {i: chr(i + 65).lower() for i in range(26)}
        
        # Game board dimensions (4x4 grid)
        self.GRID_SIZE = 4
        self.letters = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Create debug window
        cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug View", self.window_width, self.window_height)
        
        # Create event source for mouse control
        self.event_source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        
        # Load dictionary
        self.dictionary = self.load_dictionary()
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True

        # Initialize window position
        window_info = find_iphone_mirror_window()
        if not window_info:
            raise Exception("Could not find iPhone Mirror window!")

        self.window_x, self.window_y, self.window_width, self.window_height = window_info
        print(f"Window dimensions: {window_info}")

        # Save original window_y for screen capture
        self.window_y_orig = self.window_y

        # Get screen height
        screen_height = pyautogui.size()[1]

        # Adjust window_y to match Quartz coordinate system
        self.window_y = screen_height - (self.window_y + self.window_height)
        print(f"Adjusted window_y: {self.window_y}")

        # Initialize screen capture with original window_y
        self.sct = mss.mss()
        self.monitor = {
            "top": self.window_y_orig,
            "left": self.window_x,
            "width": self.window_width,
            "height": self.window_height,
            "mon": 1
        }
        print(f"Capture monitor: {self.monitor}")

        # Add a stop flag
        self.stop_flag = False
        
        # Initialize keyboard listener in a separate thread
        keyboard.on_press(self.on_press)
    
    def on_press(self, key):
        """Handle keyboard events."""
        try:
            if key.char.lower() == 'q':
                print("\nStopping solver...")
                self.stop_flag = True
        except AttributeError:
            pass  # Special key pressed, ignore it

    def detect_game_board(self, img: np.ndarray) -> Tuple[dict, np.ndarray]:
        """Detect the game board and calculate accurate board bounds."""
        debug_img = img.copy()
        
        # Crop to the region where the game board is likely to be
        height, width = img.shape[:2]
        top_crop = int(height * 0.45)  # Adjust based on your game's layout
        bottom_crop = int(height * 0.85)  # Adjust as needed
        
        cropped = img[top_crop:bottom_crop, :]
        cv2.imshow("Cropped Input", cropped)
        cv2.waitKey(1)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        cv2.imshow("Green Mask", green_mask)
        cv2.waitKey(1)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("Could not detect game board!")
        
        # Find the largest contour (assuming it's the board)
        board_contour = max(contours, key=cv2.contourArea)
        
        # Draw the selected contour
        cv2.drawContours(cropped, [board_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Selected Contour", cropped)
        cv2.waitKey(1)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(board_contour)
        print(f"Raw board bounds: x={x}, y={y}, w={w}, h={h}")
        
        # Adjust coordinates to account for cropping
        y += top_crop
        
        # Adjust for margins
        margin_left = 77   # Adjust this value to fix the left margin
        margin_right = 77  # Adjust this value to fix the right margin
        margin_top = 5     # Adjust if needed
        margin_bottom = 95  # Adjust if needed
        
        board_left = x + margin_left
        board_right = x + w - margin_right
        board_top = y + margin_top
        board_bottom = y + h - margin_bottom
        
        board_width = board_right - board_left
        board_height = board_bottom - board_top
        
        # Calculate cell sizes based on the actual board size
        cell_size_w = board_width / self.GRID_SIZE
        cell_size_h = board_height / self.GRID_SIZE
        
        # Store board bounds
        self.board_bounds = {
            'top': board_top,
            'bottom': board_bottom,
            'left': board_left,
            'right': board_right,
            'cell_size_w': cell_size_w,
            'cell_size_h': cell_size_h
        }
        
        # Calculate cell positions
        self.cell_positions = {}
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                center_x = board_left + (j + 0.5) * cell_size_w
                center_y = board_top + (i + 0.5) * cell_size_h
                self.cell_positions[(i, j)] = (
                    self.window_x + center_x,
                    self.window_y + center_y
                )
        
        # Extract board image
        board_img = img[
            int(board_top):int(board_bottom),
            int(board_left):int(board_right)
        ]
        
        print(f"Board bounds: {self.board_bounds}")
        return self.board_bounds, board_img

    def load_dictionary(self) -> Set[str]:
        """Load a dictionary of valid English words."""
        words = set()
        dictionary_path = "/usr/share/dict/words"
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if len(word) >= 3:  # Word Hunt requires words of 3+ letters
                        words.add(word)
        return words

    def capture_board(self) -> np.ndarray:
        """Capture the game board using mss."""
        # Capture the entire window first
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        
        # Show the full capture for debugging
        cv2.imshow("Full Capture", img)
        cv2.waitKey(1)
        
        # Detect and extract the game board
        _, board_img = self.detect_game_board(img)
        
        # Show the cropped board for debugging
        if board_img is not None and board_img.size > 0:
            cv2.imshow("Board Capture", board_img)
            cv2.waitKey(1)
        else:
            print("Error: Board capture is empty!")
        
        return board_img

    def preprocess_cell(self, cell_img: np.ndarray) -> np.ndarray:
        """Preprocess a cell image for better OCR."""
        # Convert to grayscale if needed
        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Initial noise reduction
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        
        # Binary threshold with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up noise and strengthen letter shapes
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Add padding (important for Tesseract)
        padding = 20
        padded = cv2.copyMakeBorder(
            binary,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        # Scale up image
        scale_factor = 3
        resized = cv2.resize(padded, None, 
                            fx=scale_factor, 
                            fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        
        return resized

    def process_image(self, img: np.ndarray) -> List[List[str]]:
        """Process the captured image using trained CNN."""
        if img is None or img.size == 0:
            print("Error: Empty image provided to process_image")
            return [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        letters = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        cell_size_w = self.board_bounds['cell_size_w']
        cell_size_h = self.board_bounds['cell_size_h']
        debug_img = img.copy()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                # Extract cell
                margin = 2
                cell_top = max(0, int(i * cell_size_h + margin))
                cell_bottom = min(img.shape[0], int((i + 1) * cell_size_h - margin))
                cell_left = max(0, int(j * cell_size_w + margin))
                cell_right = min(img.shape[1], int((j + 1) * cell_size_w - margin))
                
                cell_img = img[cell_top:cell_bottom, cell_left:cell_right]
                
                if cell_img.size == 0 or cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
                    continue
                
                # Preprocess and get prediction
                try:
                    # Preprocess cell image
                    tensor = transform(cell_img)
                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                    tensor = tensor.to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.model(tensor)
                        pred_idx = output.argmax(dim=1).item()
                        confidence = torch.exp(output.max()).item()
                        
                        letter = self.idx_to_letter[pred_idx]
                        
                        if confidence > 0.5:  # Confidence threshold
                            letters[i][j] = letter
                            print(f"Detected letter at {i},{j}: {letter.upper()} (confidence: {confidence:.2f})")
                        else:
                            print(f"Low confidence detection at {i},{j}: {letter.upper()} ({confidence:.2f})")
                
                except Exception as e:
                    print(f"Error processing cell {i},{j}: {e}")
                    continue
                
                # Debug visualization
                cv2.rectangle(
                    debug_img,
                    (cell_left, cell_top),
                    (cell_right, cell_bottom),
                    (0, 255, 0),
                    2
                )
                if letters[i][j]:
                    cv2.putText(
                        debug_img,
                        letters[i][j].upper(),
                        (cell_left + int(cell_size_w * 0.25), cell_top + int(cell_size_h * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
        
        # Show debug view
        cv2.imshow("Debug View", debug_img)
        cv2.waitKey(1)
        
        return letters

    def find_words(self) -> List[str]:
        """Find all valid words on the board using DFS."""
        def is_valid_pos(x: int, y: int) -> bool:
            return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE
        
        def dfs(x: int, y: int, visited: set, current_word: str, words: set):
            if len(current_word) >= 3 and current_word.lower() in self.dictionary:
                words.add(current_word)
            
            if len(current_word) > self.GRID_SIZE ** 2:
                return  # Prevent excessively long words
            
            # Check all 8 directions
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (is_valid_pos(new_x, new_y) and 
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    dfs(new_x, new_y, visited, 
                        current_word + self.letters[new_x][new_y], 
                        words)
                    visited.remove((new_x, new_y))
        
        found_words = set()
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.letters[i][j]:  # Only start from cells with letters
                    visited = {(i, j)}
                    dfs(i, j, visited, self.letters[i][j], found_words)
        
        return sorted(list(found_words), key=lambda x: (-len(x), x))

    def find_word_path(self, word: str) -> List[Tuple[int, int]]:
        """Find the sequence of grid positions that spell out the word."""
        def dfs(x: int, y: int, visited: set, target: str, index: int, path: list) -> bool:
            if index == len(target):
                return True
            
            # Check all 8 directions
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.GRID_SIZE and 
                    0 <= new_y < self.GRID_SIZE and 
                    (new_x, new_y) not in visited and
                    self.letters[new_x][new_y] == target[index]):
                    visited.add((new_x, new_y))
                    path.append((new_x, new_y))
                    if dfs(new_x, new_y, visited, target, index + 1, path):
                        return True
                    path.pop()
                    visited.remove((new_x, new_y))
            return False
        
        word = word.lower()
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.letters[i][j] == word[0]:
                    path = [(i, j)]
                    visited = {(i, j)}
                    if dfs(i, j, visited, word, 1, path):
                        return path
        return []

    def swipe_word(self, word: str):
        """Execute swipe motion to input a word using Quartz events."""
        path = self.find_word_path(word)
        if not path:
            print(f"Could not find path for word: {word}")
            return
        
        try:
            screen_height = pyautogui.size()[1]
            points = []
            
            # Calculate board area within window
            board_height = self.board_bounds['bottom'] - self.board_bounds['top']
            
            # Move the board down by adjusting the starting position
            board_top = self.window_y + int(self.window_height * 0.25)  # lower number = lower y offset
            board_space = self.window_height * 0.25  # Decreased from 0.4 to 0.35 for more compact board
            
            for grid_pos in path:
                i, j = grid_pos
                
                # Calculate normalized position (0-1 range)
                norm_x = (j + 0.5) / self.GRID_SIZE
                norm_y = (i + 0.5) / self.GRID_SIZE
                
                # Calculate actual screen coordinates
                board_width = self.window_width * 0.6 # Board takes up 80% of window width
                board_x_offset = self.window_width * 0.2  # 10% margin on each side
                
                # Calculate X position
                screen_x = self.window_x + board_x_offset + (norm_x * board_width)
                
                # Calculate Y position with adjusted vertical spacing
                y_pos = board_top + (norm_y * board_space)
                
                # Convert to Quartz coordinate system
                quartz_y = screen_height - y_pos
                
                print(f"Grid pos {i},{j} -> Window pos ({screen_x:.1f}, {y_pos:.1f}) -> Quartz pos ({screen_x:.1f}, {quartz_y:.1f})")
                points.append((screen_x, quartz_y))
            
            # Create event source
            source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
            
            # Mouse down at start
            start_point = CGPointMake(points[0][0], points[0][1])
            down_event = create_mouse_event(kCGEventLeftMouseDown, start_point, source)
            CGEventPost(kCGHIDEventTap, down_event)
            time.sleep(0.05)
            
            # Drag through points
            for point in points[1:]:
                quartz_point = CGPointMake(point[0], point[1])
                drag_event = create_mouse_event(kCGEventLeftMouseDragged, quartz_point, source)
                CGEventPost(kCGHIDEventTap, drag_event)
                time.sleep(0.05)
            
            # Release at end
            end_point = CGPointMake(points[-1][0], points[-1][1])
            up_event = create_mouse_event(kCGEventLeftMouseUp, end_point, source)
            CGEventPost(kCGHIDEventTap, up_event)
            
            # Wait between words
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error swiping word {word}: {e}")

    def solve(self):
        """Main solving loop."""
        # Focus window before capture
        focus_iphone_window()
        time.sleep(0.5)
        
        # Capture and process the board
        print("\nCapturing board...")
        board_img = self.capture_board()
        
        print("\nProcessing image...")
        self.letters = self.process_image(board_img)
        
        print("\nDetected letters:")
        for row in self.letters:
            print(" ".join(letter.upper() if letter else '_' for letter in row))
        
        # Verify letters are correct
        print("\nAre these letters correct? (y/n)")
        if input().lower() != 'y':
            print("Aborting...")
            return
        
        # Find all possible words
        words = self.find_words()
        print(f"\nFound {len(words)} possible words")
        
        # Sort words by length (longest first) and alphabetically
        words.sort(key=lambda x: (-len(x), x))
        
        # Execute swipes for each word
        for word in words:
            if self.stop_flag:
                print("Solver stopped by user (Q pressed)")
                break
                
            print(f"Swiping word: {word}")
            self.swipe_word(word)
            time.sleep(0.1)  # Adjust as needed

    def __del__(self):
        # Stop the keyboard listener
        if hasattr(self, 'listener'):
            self.listener.stop()
        cv2.destroyAllWindows()
    
    def test_bounds(self):
        """Move mouse around the border of the game area to visualize swipe boundaries."""
        try:
            screen_height = pyautogui.size()[1]
            
            # Calculate board area
            board_top = self.window_y + int(self.window_height * 0.25) # top margin
            board_space = self.window_height * 0.25 # vertical spacing between cells
            board_width = self.window_width * 0.6 # the width (horizontal spacing) each grid cell
            board_x_offset = self.window_width * 0.2 # where the left edge of the grid starts
            
            # Calculate corner points (clockwise from top-left)
            corners = [
                # Top-left (0,0)
                (
                    self.window_x + board_x_offset,
                    screen_height - (board_top)
                ),
                # Top-right (0,3)
                (
                    self.window_x + board_x_offset + board_width,
                    screen_height - (board_top)
                ),
                # Bottom-right (3,3)
                (
                    self.window_x + board_x_offset + board_width,
                    screen_height - (board_top + board_space)
                ),
                # Bottom-left (3,0)
                (
                    self.window_x + board_x_offset,
                    screen_height - (board_top + board_space)
                )
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

def main():
    print("\nWord Hunt Solver")
    print("1. Open the Word Hunt game")
    print("2. Press Enter to position the window")
    print("3. Options:")
    print("   T - Test boundaries")
    print("   Enter - Start solving")
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
        
        solver = WordHuntSolver()
        
        while True:
            choice = input("\nEnter choice (T/Enter/Q): ").lower()
            if choice == 't':
                solver.test_bounds()
            elif choice == 'q':
                break
            elif choice == '':
                solver.solve()
            else:
                print("Invalid choice")
        
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