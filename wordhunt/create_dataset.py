import cv2
import numpy as np
import os
import argparse
import json
import hashlib
from datetime import datetime
from wordhunt import WordHuntSolver

def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def _hash_img(img: np.ndarray) -> str:
    # Hash the raw bytes of the resized 28x28 grayscale image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    return hashlib.sha256(img.tobytes()).hexdigest()

def _load_index(output_dir: str) -> dict:
    path = os.path.join(output_dir, '.index.json')
    if not os.path.exists(path):
        # Initialize by scanning existing files to find next index per letter
        next_idx = {}
        for root, dirs, files in os.walk(output_dir):
            base = os.path.basename(root)
            if len(base) == 1 and base.isalpha():
                max_idx = -1
                for f in files:
                    if f.lower().endswith('.png') and f.upper().startswith(base.upper() + '_'):
                        try:
                            idx = int(f.split('_')[-1].split('.')[0])
                            max_idx = max(max_idx, idx)
                        except Exception:
                            pass
                next_idx[base.upper()] = max_idx + 1
        return {'next': next_idx}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {'next': {}}

def _save_index(output_dir: str, data: dict):
    path = os.path.join(output_dir, '.index.json')
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_hashes(output_dir: str) -> set:
    path = os.path.join(output_dir, '.hashes.txt')
    hashes = set()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    h = line.strip()
                    if h:
                        hashes.add(h)
        except Exception:
            pass
    return hashes

def _append_hash(output_dir: str, h: str):
    path = os.path.join(output_dir, '.hashes.txt')
    try:
        with open(path, 'a') as f:
            f.write(h + '\n')
    except Exception:
        pass

def save_cell_image(cell_img: np.ndarray, letter: str, output_dir: str, index_state: dict, dedupe_hashes: set | None = None) -> bool:
    """Save a cell image for a letter using persistent per-letter indices.

    Returns True if saved, False if skipped (e.g., duplicate).
    """
    letter = letter.upper()
    letter_dir = os.path.join(output_dir, letter)
    _ensure_dirs(letter_dir)

    # Prepare resized grayscale for saving and hashing
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    resized = cv2.resize(gray, (28, 28))

    # Dedup by content hash
    h = _hash_img(resized)
    if dedupe_hashes is not None and h in dedupe_hashes:
        print(f"[dedupe] Skipped duplicate for letter {letter}")
        return False

    # Assign next index
    next_map = index_state.setdefault('next', {})
    idx = int(next_map.get(letter, 0))
    # Find next free slot
    while True:
        filename = f"{letter}_{idx:06d}.png"
        output_path = os.path.join(letter_dir, filename)
        if not os.path.exists(output_path):
            break
        idx += 1

    # Write
    if not cv2.imwrite(output_path, resized):
        print(f"[error] Failed to save {output_path}")
        return False

    # Update index and hashes
    next_map[letter] = idx + 1
    _save_index(output_dir, index_state)
    if dedupe_hashes is not None:
        dedupe_hashes.add(h)
        _append_hash(output_dir, h)
    print(f"Saved {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create labeled letter dataset from WordHunt boards")
    parser.add_argument('--output-dir', default='letter_dataset', help='Output dataset directory')
    parser.add_argument('--no-visualize', action='store_true', help='Do not show per-cell windows')
    parser.add_argument('--suggest', action='store_true', help='Pre-fill letters using current model predictions')
    parser.add_argument('--no-dedupe', action='store_true', help='Disable duplicate detection by image hash')
    args = parser.parse_args()

    output_dir = args.output_dir
    _ensure_dirs(output_dir)

    # Persistent indices and hash dedupe store
    index_state = _load_index(output_dir)
    dedupe_hashes = None if args.no_dedupe else _load_hashes(output_dir)

    # Use same solver as main, suppress OpenCV debug windows
    solver = WordHuntSolver(no_windows=True)

    board_count = 0
    while True:
        # Capture current board
        board_img = solver.capture_board()
        if board_img is None or board_img.size == 0:
            print("[error] Empty board capture; try again.")
            continue

        # Optional suggestions using current model
        suggestions = None
        if args.suggest:
            try:
                suggestions = solver.process_image(board_img)
            except Exception as e:
                print(f"[warn] Suggestion failed: {e}")

        saved_this_board = 0
        print("\nProcessing new board...")
        for i in range(solver.GRID_SIZE):
            for j in range(solver.GRID_SIZE):
                # Extract cell ROI using the solver's current board bounds
                top = int(i * solver.board_bounds['cell_size_h'])
                bottom = int((i + 1) * solver.board_bounds['cell_size_h'])
                left = int(j * solver.board_bounds['cell_size_w'])
                right = int((j + 1) * solver.board_bounds['cell_size_w'])
                cell_img = board_img[top:bottom, left:right]

                if not args.no_visualize:
                    win = f"Cell {i},{j}"
                    cv2.imshow(win, cell_img)
                    cv2.waitKey(1)

                # Compose prompt
                sugg = None
                if suggestions and 0 <= i < len(suggestions) and 0 <= j < len(suggestions[i]):
                    sugg = suggestions[i][j]
                prompt = f"Enter letter for cell {i},{j}"
                if sugg:
                    prompt += f" (suggest: {sugg.upper()}, Enter to accept)"
                prompt += ": "

                resp = input(prompt).strip()
                if resp.lower() == 'q':
                    if not args.no_visualize:
                        cv2.destroyAllWindows()
                    print("Exiting…")
                    print(f"\nSaved {saved_this_board} cells from last board, total boards labeled: {board_count}")
                    return
                if resp == '' and sugg:
                    letter = sugg.upper()
                else:
                    letter = resp.upper()

                if len(letter) == 1 and letter.isalpha():
                    if save_cell_image(cell_img, letter, output_dir, index_state, dedupe_hashes):
                        saved_this_board += 1

                if not args.no_visualize:
                    cv2.destroyWindow(win)

        board_count += 1

        # Prompt to continue
        cont = input("\nProcess another board? (y/n): ").strip().lower()
        if cont != 'y':
            break

    if not args.no_visualize:
        cv2.destroyAllWindows()
    print(f"\nDataset created. Boards labeled: {board_count}")

if __name__ == "__main__":
    main()
