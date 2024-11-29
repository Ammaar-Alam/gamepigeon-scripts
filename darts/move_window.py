import subprocess

def move_iphone_mirror_to_corner():
    """
    Move iPhone Mirroring window to bottom right corner.
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

if __name__ == "__main__":
    move_iphone_mirror_to_corner()
