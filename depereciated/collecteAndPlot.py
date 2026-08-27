import os
import sys
import subprocess
import msvcrt  # Windows only
import time

# Resolve sibling scripts against this file's directory so the launcher works
# no matter what cwd it was started from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICO_SCRIPT = os.path.join(SCRIPT_DIR, "picoDataUnpacking.py")
VIS_SCRIPT  = os.path.join(SCRIPT_DIR, "vis.py")

for path in (PICO_SCRIPT, VIS_SCRIPT):
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Missing required script: {path}")

# === Start picoDataUnpacking.py in background ===
# Use sys.executable so the child runs in the same interpreter as the parent.
# cwd=SCRIPT_DIR makes the child's relative file writes (e.g. ppg_data.csv)
# land next to the script. stderr is inherited (default) so child errors are
# visible instead of silently swallowed.
proc = subprocess.Popen([sys.executable, PICO_SCRIPT], cwd=SCRIPT_DIR)
print("Recording... Press 'k' to stop early.")

try:
    while proc.poll() is None:
        time.sleep(0.05)             # keep keypress latency low
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # Special keys (arrows, F-keys) come as a 2-byte sequence starting
            # with 0x00 or 0xe0; consume the second byte and ignore them.
            if ch in (b'\x00', b'\xe0'):
                msvcrt.getch()
                continue
            try:
                key = ch.decode('ascii', errors='ignore').lower()
            except UnicodeDecodeError:
                continue
            if key == 'k':
                print("Detected 'k'. Stopping recording...")
                proc.terminate()
                break
except KeyboardInterrupt:
    print("KeyboardInterrupt -- stopping recording.")
    proc.terminate()

# === Wait for it to clean up (with timeout so we can't hang forever) ===
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    print("[WARN] Recorder did not exit in 5s; killing.")
    proc.kill()
    proc.wait()

# === Run analysis or postprocessing ===
print("Running next script...")
subprocess.run([sys.executable, VIS_SCRIPT], cwd=SCRIPT_DIR)
