import subprocess
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
    kCGNullWindowID,
    kCGWindowName,
    kCGWindowOwnerName,
    kCGWindowBounds
)
import time

def find_iphone_mirror_window():
    """Find the iPhone Mirror window and get its position."""
    try:
        windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID
        )
        
        for window in windows:
            name = window.get(kCGWindowName, '')
            owner = window.get(kCGWindowOwnerName, '')
            
            if 'iPhone' in str(name) or 'iPhone' in str(owner):
                bounds = window.get(kCGWindowBounds)
                if bounds:
                    print(f"Found iPhone Mirror window: {bounds}")
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
        set mirrorProcess to first process whose name contains "iPhone"
        set frontmost of mirrorProcess to true
        delay 0.5
        click menu item "Bottom Right" of menu 1 of menu item "Move & Resize" of menu 1 of menu bar item "Window" of menu bar 1 of mirrorProcess
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
        set mirrorProcess to first process whose name contains "iPhone"
        set frontmost of mirrorProcess to true
    end tell
    """
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
        time.sleep(0.1)
    except Exception as e:
        print(f"Focus failed: {e}")
