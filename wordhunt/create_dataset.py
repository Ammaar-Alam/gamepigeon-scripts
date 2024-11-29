import cv2
import numpy as np
import os
from wordhunt import WordHuntSolver

def save_cell_image(cell_img: np.ndarray, letter: str, count: int, output_dir: str):
    """Save a cell image to the appropriate directory."""
    letter_dir = os.path.join(output_dir, letter.upper())
    os.makedirs(letter_dir, exist_ok=True)
    
    # Preprocess the image
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    
    # Resize to fixed size
    resized = cv2.resize(gray, (28, 28))
    
    # Save the image
    output_path = os.path.join(letter_dir, f"{letter}_{count}.png")
    cv2.imwrite(output_path, resized)

def main():
    # Create output directory
    output_dir = "letter_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the exact same solver as the main script
    solver = WordHuntSolver()
    
    board_count = 0
    while True:
        # Capture and process board using existing methods
        board_img = solver.capture_board()
        
        print("\nProcessing new board...")
        for i in range(solver.GRID_SIZE):
            for j in range(solver.GRID_SIZE):
                # Extract cell using the same dimensions as the main solver
                cell_top = int(i * solver.board_bounds['cell_size_h'])
                cell_bottom = int((i + 1) * solver.board_bounds['cell_size_h'])
                cell_left = int(j * solver.board_bounds['cell_size_w'])
                cell_right = int((j + 1) * solver.board_bounds['cell_size_w'])
                
                cell_img = board_img[cell_top:cell_bottom, cell_left:cell_right]
                
                # Show the cell
                cv2.imshow(f"Cell {i},{j}", cell_img)
                cv2.waitKey(1)
                
                # Get correct letter from user
                letter = input(f"Enter letter for cell {i},{j} (or 'q' to quit): ").upper()
                if letter == 'Q':
                    cv2.destroyAllWindows()
                    return
                
                if len(letter) == 1 and letter.isalpha():
                    save_cell_image(cell_img, letter, board_count * 16 + i * 4 + j, output_dir)
        
        board_count += 1
        
        # Ask to continue
        if input("\nProcess another board? (y/n): ").lower() != 'y':
            break
    
    cv2.destroyAllWindows()
    print(f"\nDataset created with {board_count} boards ({board_count * 16} cells)")

if __name__ == "__main__":
    main()