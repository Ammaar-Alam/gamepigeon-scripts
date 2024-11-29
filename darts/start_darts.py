import subprocess
import webbrowser
import time
import os
import sys
import http.server
import socketserver
import threading

def start_http_server():
    """Start HTTP server to serve the HTML file"""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress logging
            pass
    
    PORT = 8000
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\nServing HTML at http://localhost:{PORT}/dartboard_feedback.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"\nPort {PORT} already in use, assuming server is running")
        else:
            raise

def main():
    print("Starting Darts AI System...")
    
    # Start HTTP server in a separate thread
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()
    
    time.sleep(1)  # Wait for server to start
    
    # Open the feedback UI in browser
    print("\nOpening feedback UI...")
    webbrowser.open('http://localhost:8000/dartboard_feedback.html')
    
    print("\nStarting AI server...")
    print("Use the browser interface to:")
    print("1. Select target number and multiplier")
    print("2. Press SPACE or click 'Throw Dart' to execute throw")
    print("3. Click where the dart landed on the board")
    print("4. Press ENTER or click 'Confirm Landing' to train AI")
    print("5. Press R or click 'Reset' to clear the landing")
    print("\nModes:")
    print("- Training: Learn and improve throws")
    print("- Competing: Use best known throws")
    print("\nUse the mode buttons in the top-right corner to switch modes.")
    print("Close this window to stop the system.")
    
    try:
        # Start the AI server
        subprocess.run([sys.executable, 'darts_feedback_ai.py'])
    except KeyboardInterrupt:
        print("\nStopping system...")

if __name__ == "__main__":
    main()
