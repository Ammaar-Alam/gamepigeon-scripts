#!/usr/bin/env python3
"""
Simulate a Word Hunt board from typed 4x4 letters (no OCR, no swiping).

Usage examples (run from the wordhunt folder):

  python simulate_board.py \
    --grid "dtio fors npat lfin" \
    --window "1160,210,344,764" \
    --targets "protistan,topiarist,antipot,patriot,artist" \
    --limit 200

Notes
- Letters must be 4 rows of 4 lowercase letters (use 'q' for QU tiles).
- The window coordinates are only used to satisfy the solver's initialization;
  no real capture or swiping occurs.
"""

import argparse
import os
import sys


def parse_window(s: str):
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 4:
        raise ValueError("--window must be 'X,Y,Width,Height'")
    x, y, w, h = map(int, parts)
    return (x, y, w, h)


def main():
    ap = argparse.ArgumentParser(description="Simulate a Word Hunt board from typed letters")
    ap.add_argument("--grid", required=True, help="4 rows of 4 letters, space-separated. Example: 'dtio fors npat lfin'")
    ap.add_argument("--window", default="1160,210,344,764", help="Bounds of the mirroring window: X,Y,Width,Height")
    ap.add_argument("--targets", default="", help="Comma-separated words to check for dictionary/path existence")
    ap.add_argument("--limit", type=int, default=200, help="Max words to print (longest first)")
    args = ap.parse_args()

    # Prepare environment to avoid any real UI/windows
    os.environ['WORDHUNT_NO_WINDOWS'] = '1'

    # Defer import until env is set
    import wordhunt

    # Monkeypatch window find/focus so the solver does not touch real windows
    bounds = parse_window(args.window)
    wordhunt.find_iphone_mirror_window = lambda: bounds
    wordhunt.focus_iphone_window = lambda: None

    # Build solver
    solver = wordhunt.WordHuntSolver(no_windows=True)

    # Parse the grid
    rows = args.grid.strip().lower().split()
    if len(rows) != 4 or any(len(r) != 4 or not r.isalpha() for r in rows):
        print("Error: --grid must be 4 rows of 4 letters (a-z), e.g. 'dtio fors npat lfin'", file=sys.stderr)
        sys.exit(2)
    solver.letters = [list(r) for r in rows]

    # Compute words
    words = solver.find_words()
    words.sort(key=lambda x: (-len(x), x))
    print(f"Found {len(words)} words")
    for w in words[: max(1, args.limit)]:
        print(w)

    # Check targets (dictionary presence and path existence on this grid)
    targets = [t.strip().lower() for t in args.targets.split(',') if t.strip()] if args.targets else []
    if targets:
        print("\nTargets on this board:")
        for t in targets:
            in_dict = t in solver.dictionary
            path = bool(solver.find_word_path(t))
            print(f"{t:10s}  in_dict={in_dict}  path={'yes' if path else 'no'}")


if __name__ == "__main__":
    main()

